# -*- coding: utf-8 -*-
import os
import shutil
import json
import pickle
import cv2
import numpy as np
from collections import OrderedDict
import itertools
import random
import copy

def sublistExists(lst, sublist):
    for i in range(len(lst)-len(sublist)+1):
        if sublist == lst[i:i+len(sublist)]:
            return True 
    return False 

# save json data function
def json_data_save(path, data):
    jsdata = json.dumps(data)
    jsfile = open(path, 'w')
    jsfile.write(jsdata)
    jsfile.close()   
    
def ExtractFrame(frame,n,gap=5,flag=False):
    frames = []
    if flag == True: # +
        for i in range(n):
            frames.append(frame+i*gap)
    else:# -
        for i in range(n):
            frames.append(frame-(n-i-1)*gap)
    return frames

# get fps of video
def GetVideoFps(video_path):
    vidcap = cv2.VideoCapture(video_path)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') # Find OpenCV version
    fps = 0
    if int(major_ver)  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
    #print(fps)
    vidcap.release()    
    return fps

def GetCaptionRelatedImageId(img_info,video_info,n):
    info = {}
    for key in img_info: #img_info[key] = ['-4RXOT_UfpM', 0, 2899, -1, path]
        img_id = int(key)
        vname = img_info[key][0]
        vindex = img_info[key][1]
        frame_id = img_info[key][2]
        caption_id = img_info[key][3]
        if vname not in info:
            info[vname] = {}
        if str(caption_id) not in info[vname]:
            info[vname][str(caption_id)] = []
        info[vname][str(caption_id)].append(img_id)
    return info    
            
    
def GetVideoCaptions(captions_path):
    video2captions = {}
    with open(captions_path, "r") as file:
        for index,f in enumerate(file):
            #读取一行，代表一个视频的所有caption
            line  = json.loads(f,object_pairs_hook=OrderedDict)
            #读取 video id
            vid = line['video_id']
            video2captions[vid] = []
            #读取包含所有steps的caption集合
            vsteps = line['step']
            for step in vsteps:
                video2captions[vid].append(vsteps[step]['caption'])
    return video2captions


'''
Input:
video_info[split] = 
    {'vid':{'frames':[],'captions':[],'areas':[]}
     ……
    }
movie_names     -Store the name of video in train/test set
caption2imgs    -Store the image id corresponding to video[vid]&caption[caption_id]
video2captions  -Store all step captions of video[vid]
n               -number of images extracted from each video clip 

Output:
data = {
          'train':{
                'mods':{‘to_str’:'apply  fundation on the face','from': [0, 1, 2, 3, 4],  'to':[1000,1001,1002,1003,1004], 'step_gap':1}
                'img_num': the number of total images(source img + target img)
                'mod_text_num': n
                'from_imgs':[]
                'to_imgs':[]
          }
          'test':{……}
      }

'''
def Tirg_taskData_prepare(video_info, movie_names, caption2imgs, video2captions, n):
    data = {}
    
    movie_names
    data = {}
    data['mods'] = []
    data['img_num'] = 0
    data['mod_text_num'] = 0
    imgs = []
    from_imgs = []
    to_imgs = []
    
    for idx,vid in enumerate(movie_names): 
        vcaps_id = video_info[vid]['captions'] #video captions index  e.g.[0,1,2,3,4,5,6,7,9]
        #print("vcaps_id",vcaps_id)
        step_num = len(vcaps_id)
        frame_num = len(video_info[vid]['frames'])
        captions = video2captions[vid]
        # Choose a random number of clips in each video
        select = 0  # select = the random number
        if step_num >= 5: #the number of steps in the video > 5
            select = 5
        elif step_num <5 and step_num >= 2:
            select = random.randint(2,step_num)
        else:
            #print(vid,"step num < 2, pass!")
            continue
                        
        combinations = list(itertools.combinations(vcaps_id,select)) 
        #Pick a random number
        iter_num = min(len(combinations),step_num)
        #print("iter_num ",iter_num )
        data['mod_text_num'] += iter_num * (select-1) #Count the total number of mod text
        cmbs = random.sample(combinations, iter_num)
        for c in cmbs: #Build an instance of training set/test set
            for i in range(select-1):
                sentence = ''
                for j in range(c[i]+1,c[i+1]+1):
                    sentence += ' '+captions[j]
                step_gap = c[i+1]-c[i]
                
                mod = {}
                mod['to_str'] = sentence
                mod['from'] = caption2imgs[vid][str(c[i])]
                mod['to'] = caption2imgs[vid][str(c[i+1])]
                mod['step_gap'] = step_gap
                data['mods'].append(mod)
                imgs.extend(mod['from'])
                imgs.extend(mod['to'])
                to_imgs.extend(mod['to'])
                from_imgs.extend(mod['from'])
                    
    data['img_num'] = len(set(imgs))
    data['to_imgs'] = list(set(to_imgs))
    data['from_imgs'] = list(set(from_imgs))
    
    return data   


if __name__ == '__main__':
    img_file_path = {}
    img_file_path['train'] = "../shared_data/train_images/"
    img_file_path['test'] = "../shared_data/val_images/"
    #load split movie names
    movie_names = {}
    movie_names['train'] = np.load("../shared_data/train_vids.npy")
    movie_names['test'] = np.load("../shared_data/test_vids.npy")
    #load image information
    fs = open("../shared_data/img_index2info.json",'r')
    img_info = json.load(fs) 
    fd = open("../shared_data/video_info.json",'r')
    video_info = json.load(fd)
    
    # n is the num of extracted images for each video clip aligned with a makeup step
    n = 10
    #path of caption annotations
    captions_path = {}
    captions_path['train'] = "../../../YouMakeup/data/train/train_steps.json"
    captions_path['test'] = "../../../YouMakeup/data/valid/valid_steps.json"

    video2captions = {}
    caption2imgs = {}
    data = {}

    for split in ['train','test']:
        video2captions[split] = GetVideoCaptions(captions_path[split]) #save all step captions of each video
        caption2imgs[split] = GetCaptionRelatedImageId(img_info[split],video_info[split],n)
        data[split]  = copy.deepcopy(Tirg_taskData_prepare(video_info[split], movie_names[split], caption2imgs[split], video2captions[split], n))
    
    #save data
    save_path = 'mod_data.json'
    json_data_save(save_path, data) 

