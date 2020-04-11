import numpy as np
import json
import h5py
import string
import random
import unicodedata
import os

train_caption_path = '../../../YouMakeup/data/train/train_steps.json' 
dev_caption_path = '../../../YouMakeup/data/valid/valid_steps.json'

def json_read(json_path):
    file = open(json_path, 'r')
    items = []
    for line in file.readlines():
        dic = json.loads(line)
        items.append(dic)
    return items

def generate_train_captions(data_path):
    train_caption_all = json_read(train_caption_path)
    train_list = []

    for item in train_caption_all:
        steps = item['step']
        
        for i in range(len(steps)-1):
            now_sentence = steps[str(i+1)]['caption']           
            j = i+2
            while j <= len(steps):
                next_sentence = steps[str(j)]['caption']
                train_item = [now_sentence,next_sentence,1]
                train_list.append(train_item)
                j += 1 
        
    print("training  data processed successfully.")
    print("start shuffling......")
    random.shuffle(train_list)
    print("shuffle finished,start generate negatative labels")

    for i in range(int(len(train_list)/2)):
        train_list[-1*i][0],train_list[-1*i][1] = train_list[-1*i][1],train_list[-1*i][0]
        train_list[-1*i][2] = 0
    random.shuffle(train_list)
    
    print("negative samples generate successfully, shuffle training data again")
    
    train_sentence1_list = []
    train_sentence2_list = []
    train_label_list = []
    for item in train_list:
        train_sentence1_list.append(unicodedata.normalize('NFKD', item[0]).encode('ascii','ignore'))
        train_sentence2_list.append(unicodedata.normalize('NFKD', item[1]).encode('ascii','ignore'))
        train_label_list.append(item[2])
            
    print("save h5......")
    train_h5=h5py.File(data_path+"/train_data.h5","w")
    train_h5['sentence_1'] = np.string_(train_sentence1_list)
    train_h5['sentence_2'] = np.string_(train_sentence2_list)
    train_h5['label'] = np.array(train_label_list)
    
    print("**********finised.**********")
    print("train_len:",len(train_list))
    
def generate_dev_captions(data_path):
    dev_caption_all = json_read(dev_caption_path)
    dev_dict = {}
    for item in dev_caption_all:
        steps = item['step']
        
        for i in range(len(steps)-1):
            now_sentence = steps[str(i+1)]['caption']           
            j = i+2
            while j <= len(steps):
                if j-i-1 <= 5:
                    key = j-i-1
                else:
                    key = 5
                if key not in dev_dict.keys():
                    dev_dict[key] = []
                next_sentence = steps[str(j)]['caption']
                dev_item = [now_sentence,next_sentence,1]
                dev_dict[key].append(dev_item)
                j += 1 
        
    print("dev data processed successfully.")
    print("start shuffling......")
    for key in dev_dict.keys():
        random.shuffle(dev_dict[key])
    print("shuffle finished,start generate negatative labels")

    train_list_all = []
    for key in dev_dict.keys():
        len_dev_dict = len(dev_dict[key])
    
        for i in range(int(len_dev_dict/2)):
            dev_dict[key][-1*i][0],dev_dict[key][-1*i][1] = dev_dict[key][-1*i][1],dev_dict[key][-1*i][0]
            dev_dict[key][-1*i][2] = 0  

    print("negative samples generate successfully")
    
    dev_sentence1_all = []
    dev_sentence2_all = []
    dev_label_all = []
    for key in dev_dict.keys():
        dev_sentence1_list = []
        dev_sentence2_list = []
        dev_label_list = []
           
        for item in dev_dict[key]:
            dev_sentence1_list.append(unicodedata.normalize('NFKD', item[0]).encode('ascii','ignore'))
            dev_sentence2_list.append(unicodedata.normalize('NFKD', item[1]).encode('ascii','ignore'))
            dev_label_list.append(item[2])
        
        dev_sentence1_all = dev_sentence1_all + dev_sentence1_list
        dev_sentence2_all = dev_sentence2_all + dev_sentence2_list
        dev_label_all = dev_label_all + dev_label_list
       
        print("save h5......")
        
        dev_h5=h5py.File(data_path+"/dev_data_"+str(key)+".h5","w")
        dev_h5['sentence_1'] = np.string_(dev_sentence1_list)
        dev_h5['sentence_2'] = np.string_(dev_sentence2_list)
        dev_h5['label'] = np.array(dev_label_list)
        
    dev_h5_all=h5py.File(data_path+"/dev_data.h5","w")
    dev_h5_all['sentence_1'] = np.string_(dev_sentence1_all)
    dev_h5_all['sentence_2'] = np.string_(dev_sentence2_all)
    dev_h5_all['label'] = np.array(dev_label_all)

    print("**********finished.**********")
    for key in dev_dict.keys():
        print('step:',key," dev_len:",len(dev_dict[key]))



if __name__ == '__main__':
    data_path = './data'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    generate_train_captions(data_path)#choose 0.2 as dev dataset
    generate_dev_captions(data_path)
    
    

