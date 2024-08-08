# implementation for **DSL** in active source free open-set domain adaptation
 

### Prerequisites:
- python == 3.7.3
- pytorch == 1.0.1.post2
- torchvision == 0.2.2
- numpy, scipy, sklearn, PIL, argparse, tqdm

### Dataset:

- Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) from the official websites, and modify the path of images in each '.txt' under the folder './object/data/'. The source model of office-home and visda can be downloaded in this [Url](https://drive.google.com/drive/folders/1eiJtky4seNApOSYJiGrDywfJbCBp_3sb)


### Training:
	
##### Active Source free open-set domain adaptation (ASFODA) on the dataset
	- Train model on the source domain **A** (**s = 0**), we view the full source data as a test set.
    ```python
    cd object/
    python image_source.py --trte full --da oda --output ckps/source/ --gpu_id 0 --dset office --max_epoch 100 --s 0 --t 1
    ```
	
	- Adaptation to other target domains **D and W**, respectively
    ```python
    python image_target_ASFODA_DSL.py  --ratio 0.05 --lada 0.5 --da oda --output_src ckps/source/ --output ckps/target/ --gpu_id 0 --dset office --s 0 --t 1  
    ```

	
