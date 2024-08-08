import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from common.utils.analysis import collect_feature_id, tsne
from common.utils.data import ForeverDataIterator
from loss import CrossEntropyFeatureAugWeight, Cross_Entropy_Open_T

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        new_all_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split('\t')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                    new_all_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
                    new_all_tar.append(reci[0] + ' ' + str(int(reci[1])) + '\n')

        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
        txt_all = new_all_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["tsne"] = ImageList_idx(txt_all, transform=image_test())
    dset_loaders["tsne"] = DataLoader(dsets["tsne"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC,netCU, flag=False,cls_unknown=[]):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            outputs_un = netCU(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_output_un = outputs_un.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_output_un = torch.cat((all_output_un, outputs_un.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    _, predict_un = torch.max(all_output_un, 1)
    predict_un = cls_unknown[predict_un]

    if True:
        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)

        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        labels = kmeans.predict(ent.reshape(-1,1))

        idx = np.where(labels==1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        predict[np.where(labels==iidx)[0]] = args.class_num
        
        all_label_hos = copy.deepcopy(all_label)
        all_label_hos[all_label_hos >= args.class_num] = args.class_num
        matrix = confusion_matrix(all_label_hos, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label_hos).astype(int),:]

        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        unknown_acc = acc[-1:].item()
        hos = (2 * np.mean(acc[:-1]) *unknown_acc) / ( np.mean(acc[:-1])  + unknown_acc)

        unknwon_true_pre = predict_un[all_label >= args.class_num]
        unknown_true_lbl = all_label[all_label >= args.class_num]
        new_cls_unknown = np.sum(unknwon_true_pre == unknown_true_lbl.float().numpy()) / len(unknown_true_lbl)
        print(new_cls_unknown)

        NP = len(cls_unknown) / args.class_num_un

        nos = (2 * NP *new_cls_unknown) / ( NP + new_cls_unknown)

        return np.mean(acc[:-1]), np.mean(acc), unknown_acc,hos,new_cls_unknown * 100,nos * 100

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s
def data_load_o31(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        tar_spilt_cls = []
        for index in range(len(args.tar_classes)):
            if index in args.src_classes:
                tar_spilt_cls.append(index)
            elif index >=20 and index < 31:
                tar_spilt_cls.append(index)

        new_tar = []
        new_all_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                    new_all_tar.append(line)
                elif  int(reci[1]) in tar_spilt_cls:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
                    new_all_tar.append(reci[0] + ' ' + str(int(reci[1])) + '\n')

        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
        txt_all = new_all_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)
    dsets["tsne"] = ImageList_idx(txt_all, transform=image_test())
    dset_loaders["tsne"] = DataLoader(dsets["tsne"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders
def train_target(args):
    if args.dset == 'office':
        dset_loaders = data_load_o31(args)
    else:
        dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(args.modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    tt = 0
    iter_num = 0
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval

    flag = 0 
    h_dict = []

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue
        if iter_num / interval_iter == 0:
            flag = 0
        else:
            flag = 2
      

        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            mem_label, mem_weight,flag_cls,h_dict,cls_unknown= obtain_label(dset_loaders['tsne'], netF, netB, netC, args,flag,h_dict)
            mem_label = torch.from_numpy(np.array(mem_label)).cuda()
            mem_weight = torch.from_numpy(np.array(mem_weight)).cuda()
            flag_cls = torch.tensor(flag_cls).cuda()
            netF.train()
            netB.train()
            if iter_num == 0:
                netCU = network.feat_classifier(type=args.layer, class_num = len(cls_unknown), bottleneck_dim=args.bottleneck).cuda()
                param_group_U = []
                for k, v in netCU.named_parameters():
                    param_group_U += [{'params': v, 'lr': args.lr * args.lr_decay1}]
                optimizerU = optim.SGD(param_group_U)
                optimizerU = op_copy(optimizerU)
                netCU.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        pred = mem_label[tar_idx]
        weight = mem_weight[tar_idx]
        flag_cls_ba = flag_cls[tar_idx]

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        #common data
        outputs_test_known = outputs_test[flag_cls_ba == 1, :]
        pred_kn = pred[flag_cls_ba == 1]
        weight_kn = weight[flag_cls_ba == 1]

        #public data
        outputs_test_unknown = outputs_test[flag_cls_ba == 2, :]
        pred_un = pred[flag_cls_ba == 2]
        alpha = np.float64(2.0 / (1.0 + np.exp(-iter_num/ max_iter)) - 1.0) + args.lada
        classifier_loss =  Cross_Entropy_Open_T(num_classes=args.class_num)(outputs_test_unknown,pred_un)
        classifier_loss = alpha * classifier_loss

        #public data netCU training
        outputs_un_net = netCU(features_test)
        outputs_un_net_unknown = outputs_un_net[flag_cls_ba == 2, :]
        pred_un_net = pred[flag_cls_ba == 2]
        weight_un = weight[flag_cls_ba == 2]
        classifier_loss_un = CrossEntropyFeatureAugWeight(num_classes=len(cls_unknown))(outputs_un_net_unknown, pred_un_net.cuda(),weight_un.cuda()).float()
        # print(classifier_loss_un)


        flag_pred = True
        if len(pred_kn) == 0:
            print(tt)
            del features_test
            del outputs_test
            tt += 1
            flag_pred = False
            continue
        if flag_pred == True:
            classifier_loss_kn = CrossEntropyFeatureAugWeight(num_classes=args.class_num)(outputs_test_known, pred_kn.cuda(),weight_kn.cuda()).float()
            classifier_loss += classifier_loss_kn
            if args.ent:
                softmax_out_known = nn.Softmax(dim=1)(outputs_test_known)
                entropy_loss = torch.mean(loss.Entropy(softmax_out_known))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss -= gentropy_loss
                classifier_loss += entropy_loss * args.ent_par
        classifier_loss += classifier_loss_un

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netCU.eval()
            acc_os1, acc_os2, acc_unknown,hos,new_cls_unknown, nos = cal_acc(dset_loaders['tsne'], netF, netB, netC,netCU, True,cls_unknown)            
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}% / {:.2f}% / {:.2f}%, new classes {}/{:.2f}%,nos {:.2f}%'.format(args.name, iter_num, max_iter, acc_os2, acc_os1, acc_unknown,hos,len(cls_unknown),new_cls_unknown,nos)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()
            netCU.train()
    
    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC


def obtain_label(loader_tsne, netF, netB, netC, args,flag,h_dict):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader_tsne)
        for _ in range(len(loader_tsne)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    # con, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()

    
    fir_max, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    # more smaller, more consideration
    max2 = torch.topk(all_output, k=2, dim=1, largest=True).values
    BVSB_uncertainty = max2[:, 0] - max2[:, 1]
    # more higher, more consideration
    con = 1 - fir_max
    if args.standard == 'con':
        stand = (-con)
    elif args.standard == 'bvsb':
        stand = BVSB_uncertainty
    elif args.standard == 'ent':
        stand = (-ent)
    sor = np.argsort(stand)
    


    active_num = int(len(predict) * args.ratio)
    c_k = args.ck
   
  
    distance_all = all_fea @ all_fea.t()
    _, idx_near_all = torch.topk(distance_all, dim=-1, largest=True, k=c_k)

    from sklearn.cluster import KMeans
    if flag == 0:

        h_dict = {}
        for index in range(100):
            h_dict[index] = []
        
        
        all_sel_loc = []
        index = 0
        print(len(all_sel_loc))
        while index < active_num:
            idx_cls = sor[index]
            idx_true_lb = all_label[idx_cls].int().tolist()
            h_dict[idx_true_lb].append(idx_cls.tolist())
            all_sel_loc.append(idx_cls)
            index = index + 1
        
           
    final_sel_loc = []
    final_sel_label = []
    for index in range(len(h_dict)):
        if len(h_dict[index]) == 0:
                continue
        else:
            for i_idx in range(len(h_dict[index])):
                re_can_id = h_dict[index][i_idx]
                final_sel_loc.append(re_can_id)
                final_sel_label.append(index)
    print(np.sum(np.array(final_sel_label) == all_label[final_sel_loc].float().numpy()) / len(final_sel_loc))
    print(len(final_sel_loc))

    known_oracle_idx = np.array(final_sel_loc)[np.where(np.array(final_sel_label) < args.class_num)[0]]
    unknown_oracle_idx = np.array(final_sel_loc)[np.where(np.array(final_sel_label) >= args.class_num)[0]]

    
    kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
    labels = kmeans.predict(ent.reshape(-1,1))

    idx = np.where(labels==1)[0]
    iidx = 0
    if ent[idx].mean() > ent.mean():
        iidx = 1
    known_idx = np.where(kmeans.labels_ != iidx)[0]

    true_known_idx = []
    for index in range(len(known_idx)):
        can_idx = known_idx[index]
        if can_idx in unknown_oracle_idx:
            continue
        else:
            true_known_idx.append(can_idx)


    for index in range(len(known_oracle_idx)):
        can_idx = known_oracle_idx[index]
        if can_idx not in true_known_idx:
            true_known_idx.append(can_idx)
            
    true_known_idx = np.array(true_known_idx)

    true_unknown_idx = []
    for index in range(len(predict)):
        if index not in true_known_idx:
            true_unknown_idx.append(index)
    true_unknown_idx = np.array(true_unknown_idx)


    all_fea_kn = all_fea[true_known_idx,:]
    all_output = all_output[true_known_idx,:]
    predict = predict[true_known_idx]
    all_label_idx = all_label[true_known_idx]


    all_fea_kn = all_fea_kn.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea_kn)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea_kn, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea_kn)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea_kn, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    guess_label = args.class_num * np.ones(len(all_label), )
    guess_label[true_known_idx] = pred_label

    acc = np.sum(guess_label == all_label.float().numpy()) / len(all_label_idx)


    all_fea_un = all_fea[true_unknown_idx,:]
    initc_un = []
    cls_unknown = []
    for index in range(len(h_dict)):
        can_data = h_dict[index]
        if index >= args.class_num and len(can_data) != 0:
            cls_unknown.append(index)
            can_data_near = idx_near_all[can_data]
            fea_cls =  all_fea[can_data_near]
            initc_un.append(torch.mean(torch.mean(fea_cls,dim=1),dim=0).tolist())
    
    initc_un = np.array(initc_un)
    dd_un = cdist(all_fea_un, initc_un, args.distance)
    pred_label_un = dd_un.argmin(axis=1)
    cls_unknown = np.array(cls_unknown)
    #
    print("unknown class cluster acc:")
    print(np.sum(cls_unknown[pred_label_un]== all_label[true_unknown_idx].float().numpy()) / len(all_label[true_unknown_idx]))
   


    weight = []
    pred_one = []
    flag_cls = []

    for index in range(len(guess_label)):
        label_one = args.class_num
        if index in true_unknown_idx:
            weight.append(1.0)
            flag_cls.append(2)
            label_one = pred_label_un[true_unknown_idx.tolist().index(index)]
        elif index in final_sel_loc:
            flag_cls.append(1)
            label_one = final_sel_label[final_sel_loc.index(index)]
            weight.append(1.0)
        else:
            flag_cls.append(1)
            label_one = int(guess_label[index])
            weight.append(args.cls_par)
        pred_one.append(label_one)

    return pred_one, weight, flag_cls,h_dict,cls_unknown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office-ho me','office'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--ck', type=int, default=15, help="neighbor numbers")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lada', type=float, default=0.5)
    parser.add_argument('--ratio', type=float, default=0.05)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])

    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='./Others/office-home')
    parser.add_argument('--output_src', type=str, default='./TSM/office-home')
    parser.add_argument('--standard', type=str, default='ent', choices=['con', 'ent','bvsb'])
    parser.add_argument('--da', type=str, default='oda', choices=['oda'])
    parser.add_argument('--issave', type=bool, default=False)
    args = parser.parse_args()
       
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_world']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

   

    folder = './datasets/data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = args.t_dset_path

    if args.dset == 'office-home':
        if args.da == 'oda':
            args.class_num = 25
            args.class_num_un = 40
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]
    if args.dset == 'office':
        if args.da == 'pda':
            args.class_num = 21
            args.src_classes = [i for i in range(21)]
            args.tar_classes = [i for i in range(10)]
        if args.da == 'oda':
            args.class_num = 10
            args.class_num_un = 11
            args.src_classes = [i for i in range(10)]
            args.tar_classes = [i for i in range(31)]

    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
    args.name = names[args.s][0].upper()+names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'Other_cls_par_alpha_' + str(args.cls_par)+'_standard'+ args.standard + str(args.lada)+'_ratio_'+ str(args.ratio)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)