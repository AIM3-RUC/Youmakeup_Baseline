# -*- coding: utf-8 -*-
'''
input: videos
output: extracted images 
      format: ".jpg" & ".pt"
      Note: extract 10 frames from each video clip aligned with a makeup step in all videos
'''
import os
import shutil
import json
import pickle
import cv2
import numpy as np
from collections import OrderedDict
import PIL
from PIL import Image
import torch
import torchvision
from tqdm import tqdm


def TimeToFrameIndex(t,fps, max_frame):
    #the format of input time e.g.00:02:22
    hour = int(t[0:2])
    minute = int(t[3:5])
    second = int(t[6:8])
    frame_index = int((second + minute*60 + hour*60*60)* fps)
    if frame_index > max_frame:
        frame_index = max_frame
    return frame_index

def ExtractFrame(frame,n,gap=5,flag=False):
    frames = []
    if flag == True: # +
        for i in range(n):
            frames.append(frame+i*gap)
    else:# -
        for i in range(n):
            frames.append(frame-(n-i-1)*gap)
    return frames

# save json data function
def json_data_save(path, data):
    jsdata = json.dumps(data)
    jsfile = open(path, 'w')
    jsfile.write(jsdata)
    jsfile.close()       
    
# get fps of video
def GetVideoFps(video_path):
    vidcap = cv2.VideoCapture(video_path)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') # Find OpenCV version
    fps = 0
    if int(major_ver)  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
    FrameNumber = int(vidcap.get(7))-1
    vidcap.release()    
    return fps,FrameNumber

#convert img to the tensor format
def ConvertToTensor(image):
    transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    image = image.convert('RGB')
    img = transform(image)
    return img

def GetSplitVideoName(train_id_path,test_id_path,train_videos_path,test_videos_path):
    f = open(train_id_path, "r")
    train_ids = []
    for x in f:
        train_ids.append(x[:-1])
        
    f = open(test_id_path, "r")
    test_ids = []
    for x in f:
        test_ids.append(x[:-1])  
    #maybe some videos are missing(can't be downloaded),filter these video ids
    exist_train_videos = os.listdir(train_videos_path) #".mp4 foramt"
    exist_test_videos = os.listdir(test_videos_path)
    exist_train_videos = [v[:-4] for v in exist_train_videos]
    exist_test_videos = [v[:-4] for v in exist_test_videos]
    train_ids = list( set(exist_train_videos) & set(train_ids) )
    test_ids = list( set(exist_test_videos) & set(test_ids) )
    return train_ids,test_ids

'''
input：
---- captions_path
---- videos_path
---- movie_names
---- save_path
---- n

output：
----img_index2info : save a img id related info:[video id，video index，frame index,caption index,saved path]
----video_info     : save processing information of each video, including extracted frame ids，effective caption ids
    {'vid':
            {
                'frames':   list() type
                'captions': list() type
                'areas':    list() type
            }
     ……
    }
----split_images_num : the total number of images in train/test set



'''

def ExtractImagesFromFullData(caption_data_path,video_data_path,movie_names,save_img_path,split,n):
    img_index2info = {}
    video_info = {}
    videos_fps = []
    
    vnum = 0  #the processed video num
    image_index = 0
    gap = 5   #the interval of two extracted frames
  
    with open(caption_data_path, "r") as file:
        for index,f in enumerate(file):
            img_frame2capt_id = {}
            #read a line，include all caption annotations of a video
            line  = json.loads(f,object_pairs_hook=OrderedDict)
            #read video id
            vid = line['video_id']
            #filter video does't exist
            if vid not in movie_names: 
                continue
            video_info[vid] = {}
            video_info[vid]['areas'] = []
            areas = []
            #get fps of the video
            this_video_path = video_data_path + vid + ".mp4"
            fps,max_frame = GetVideoFps(this_video_path)
            videos_fps.append(fps)
                    
            #get captions set of the video
            vsteps = line['step']
                    
            #calculate the frame index of images to be extracted
            extract_frame_index = []
            step_period = {}
                      
            vnum += 1
            cur_frame_index = []
            first_idx = 0
            for idx,key in enumerate(vsteps):
                #each caption noted as d
                d  = vsteps[key]
                caption = d['caption']
                startime = d['startime']
                endtime = d['endtime']
                areas.append(d['area'])
                #convert timestamp to frame index
                startframe = TimeToFrameIndex(startime,fps,max_frame)
                endframe = TimeToFrameIndex(endtime,fps,max_frame)
                duration = endframe - startframe
                        
                #filter steps that are wrong annotated
                if endframe <= startframe:
                    #print(vid,"caption ",idx," error:endframe <= startframe",startime,endtime)
                    continue
                            
                if duration>int(10*fps): #if duration video clip>10s，extract a frame per 5 frames
                    gap = 5
                else: # <10s, extract frames continuously 
                    gap = 1
                        
                cur_frame_index = ExtractFrame(endframe,n,gap,False)
                extract_frame_index.extend(cur_frame_index)
                step_period_name = vid + "_" + str(startframe) + "_" + str(endframe)
                for f in cur_frame_index:
                    img_frame2capt_id[f] = idx
                    step_period[f] = step_period_name

            #-------------------------------------read videos------------------------------------
            #get extracted image useful info
            video_info[vid]['frames'] = []
            video_info[vid]['captions'] = set()
            
            this_video_path = video_data_path + vid + ".mp4"
            video = cv2.VideoCapture(this_video_path)
            times = 0  #frame index
            while True:
                res, image = video.read()
                if not res:       
                    print(index," finish reading video ",vid)
                    break
                if times in extract_frame_index:   #this frame need to be extracted
                    cid = img_frame2capt_id[times] #this frame belongs to caption cid
                    image_dir = step_period[times] # e.g. 3L-C3lsV2_w_6813_7819
                    img_file_path = save_img_path+image_dir+'/'
                    if not os.path.exists(img_file_path):
                        os.makedirs(img_file_path)
                    
                    #save img as jpg
                    output_image_path = img_file_path + vid + "_" + str(times) + ".jpg" #e.g.3L-C3lsV2_w_6813_7819/3L-C3lsV2_w_7819.pt
                    cv2.imwrite(output_image_path, image)
                    
                    #convert and save img to tensor format
                    output_image_tensor_path = img_file_path + vid + "_" + str(times) + ".pt"
                    img_tensor = ConvertToTensor(image)
                    torch.save(img_tensor, output_image_tensor_path)
                    
                    #obtain each extracted image information
                    path = image_dir+'/'+vid+'_'+str(times)+'.pt' #3L-C3lsV2_w_6813_7819/3L-C3lsV2_w_7819.pt
                    img_index2info[image_index] = [vid,vnum-1,times,cid, path]
                    video_info[vid]['frames'].append(times)
                    video_info[vid]['captions'].add(cid)
                    image_index += 1
                    
                    if times == max(extract_frame_index):
                        print(index," finish reading video ",vid)
                        break
                times += 1    
            video.release()
            video_info[vid]['captions'] = list(video_info[vid]['captions'])
            video_info[vid]['areas'] = [areas[video_info[vid]['captions'][i]] for i in range(0,len(video_info[vid]['captions'])-1)]
    
    print("**********************Finished**********************")
    print(split)
    print(vnum," videos processed in total")    
    print(image_index," images extracted in total")
    return  img_index2info,video_info,split_images_num,videos_fps  



if __name__ == '__main__':
    
    #video saved path
    videos_path = {}
    videos_path['train'] = "../../YouMakeup/data/train/videos/"
    videos_path['test'] = "../../YouMakeup/data/valid/videos/"
    #path of caption annotations
    captions_path = {}
    captions_path['train'] = "../../YouMakeup/data/train/train_steps.json"
    captions_path['test'] = "../../YouMakeup/data/valid/valid_steps.json"
    #the save path of extracted train/test imgs
    save_img_path = {}
    save_img_path['train'] = "./shared_data/train_images/"
    save_img_path['test'] = "./shared_data/val_images/"
    #get train/test video ids (filter videos that doesn't exist)
    train_id_path = "../../YouMakeup/data/train/train_id"
    test_id_path = "../../YouMakeup/data/valid/valid_id"
    movie_names = {}
    movie_names['train'],movie_names['test'] = GetSplitVideoName(train_id_path,test_id_path,videos_path['train'],videos_path['test'])
    #save train/test movie_names
    np.save("./shared_data/train_vids.npy",movie_names['train'])
    np.save("./shared_data/test_vids.npy",movie_names['test'])
    
    
    n = 10 #each video clip extract 10 frames
    img_index2info = {}
    video_info = {}
    split_images_num = {}
    fps = {}
    for split in ['train','test']:
        img_index2info[split],video_info[split],split_images_num[split], fps[split] = ExtractImagesFromFullData(captions_path[split],videos_path[split],movie_names[split],save_img_path[split],split,n)
    
    #save useful information about extracted img 
    #img_index2info,video_info
    json_data_save("./shared_data/img_index2info.json", img_index2info)
    json_data_save("./shared_data/video_info.json", video_info)
    json_data_save("./shared_data/fps.json", fps)
    
    
    
    
    
    
    
    
    
    