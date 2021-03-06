# Prepare image ordering data

To reproduce two baseline models,  you need to prepare the required data in advance.



## Requirement

- python 2.7
- Pytorch 0.4.1
- opencv



## Data download

- Download [YouMakeup dataset](https://github.com/AIM3-RUC/YouMakeup.git) and place it in the root dictionary. The file structure under the root is as followings:

  ```csharp
  ├─YouMakeup
  │  └─data
  │      ├─task
  │      ├─train
  │      └─valid
  └─Youmakeup_Baseline
      └─image_ordering
         ├─Pairwise
         ├─shared_data
         └─TIRG
  ```
  
  
  
- Download  1,680 train videos and 280 validation videos under the guidance of   [YouMakeup/data](https://github.com/AIM3-RUC/YouMakeup/tree/master/data) and place them at   ```/YouMakeup/data/train/videos/``` and  ```/YouMakeup/data/valid/videos/```  respectively

  

## Data pre-processing 

To train/evaluate two baseline models, firstly extract images from obtained videos. For each video, we extract 10 frames at the end of each video clip aligned with a  makeup step caption. Run the following code to extract images:

```
python ./shared_data/data_preprocess.py
```

- The extracted images will be put in ```./shared_data/train_images``` and ``` ./shared_data/val_images``` 

- Each image name means

    ```"VideoId_StepStartFrame_StepEndFrame / VideoId_FrameIndex.pt"```
  
   e.g. "9XP1Vs9Hz4E_7576_7864/9XP1Vs9Hz4E_7819.pt"

- The code will generate three JSON files saving useful information about extracted images in  ```./shared_data/```  after running.

## Get Image Features (Optional)

For facial image ordering task, we provide 512D  feature embedding of ResNet-18 pretrained on ImageNet and fine-tuned on YouMakeup dataset. 

If necessary, you can download the ResNet-18 embedding for train/dev images and for images in task questions from [Google Drive](https://drive.google.com/drive/folders/1JyhsQtftSXs5n46hutRMeJudopW38bET?usp=sharing) or [Baidu yun [Fetch Code: ynwa]](https://pan.baidu.com/s/1zuk7ENAWa0aMGjRQFwsjKg). 

