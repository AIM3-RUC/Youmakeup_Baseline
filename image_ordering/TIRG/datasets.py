#coding:utf-8

# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Provides data for training and testing."""
import numpy as np
import PIL
import skimage.io
import torch
import json
import torch.utils.data
import torchvision
import warnings
import random
import os


from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX,self).__iter__())
    

class BaseDataset(torch.utils.data.Dataset):
  """Base class for a dataset."""

  def __init__(self):
    super(BaseDataset, self).__init__()
    self.imgs = []
    self.test_queries = []

  def get_loader(self,
                 batch_size,
                 shuffle=False,
                 drop_last=False,
                 num_workers=0):
    return DataLoader(
        self,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=lambda i: i)

  def get_test_queries(self):
    return self.test_queries

  def get_all_texts(self):
    raise NotImplementedError

  def __getitem__(self, idx):
    return self.generate_random_query_target()

  def generate_random_query_target(self):
    raise NotImplementedError

  def get_img(self, idx, raw_img=False):
    raise NotImplementedError



class YouMakeup(BaseDataset):
  """YouMakeup dataset."""
  def __init__(self, opt, path, split='train', transform=None):
    super(YouMakeup, self).__init__()
    #path = ../shared_data/
    self.split_file = {'train':'train_images/','test':'val_images/'}
    self.img_tensor_path = path + self.split_file[split]
    self.transform = transform
    self.split = split
    f = open("mod_data.json",'r')
    self.data = json.load(f)
    self.mods = self.data[self.split]['mods']
    fs = open(path+"img_index2info.json",'r')
    self.img_info = json.load(fs)[self.split]
    
    self.source_imgs = []
    self.target_imgs = []
    self.to_img_ids = []
    self.all_from_img_ids = self.data[self.split]['from_imgs']
    self.all_to_img_ids = self.data[self.split]['to_imgs']
    self.img_num = self.data[self.split]['img_num']
    self.query_base = {'train':5, 'test':1}
    self.generate_test_queries_()
    

  def generate_test_queries_(self):
    test_queries = {}
    target_imgs = {}
    to_img_ids = {}
    for idx,mod in enumerate(self.mods):
        i = mod['from'][-1]
        j = mod['to'][-1]
        vid = self.img_info[str(i)][0]
        if vid not in test_queries.keys():
            test_queries[vid] = []
        if vid not in target_imgs.keys():
            target_imgs[vid] = []
        if vid not in to_img_ids.keys():
            to_img_ids[vid] = []
        target_imgs[vid] += [{
          'img_id':j,
          'source_img': i,
          'captions': [str(idx)]  
        }] 
        to_img_ids[vid].append(j)
        test_queries[vid] += [{
            'source_img_id': i,
            'target_img_id': j,
            'target_caption': [str(idx)],
            'mod': {
                'str': mod['to_str'] #modify text/caption
            }
        }]
    #delete videos in which target imgs num < 5
    del_key = []
    for key in to_img_ids:
        to_img_ids[key] = list(set(to_img_ids[key]))
        if len(to_img_ids[key]) < 5:
            del_key.append(key)
    
    for key in del_key:
        del to_img_ids[key]
        del test_queries[key]
        del target_imgs[key]
    self.test_queries = test_queries
    self.target_imgs = target_imgs
    self.to_img_ids = to_img_ids

  def __getitem__(self, idx):
    i = np.random.randint(0, len(self.mods)) 
    mod = self.mods[i]
    query_num = len(mod['to'])
    j = np.random.randint( query_num-5 , query_num)
    img1id, modid, img2id = mod['from'][j], i, mod['to'][j]
    out = {}
    out['source_img_id'] = img1id
    out['source_img_data'] = self.get_img(img1id)
    out['target_img_id'] = img2id
    out['target_img_data'] = self.get_img(img2id)
    out['mod'] = {'id': modid, 'str': self.mods[modid]['to_str']}
    return out
        
        
  def __len__(self):
    return len(self.mods)*self.query_base[self.split]

  def get_all_texts(self):
    return [mod['to_str'] for mod in self.mods]

  def get_img(self, idx, raw_img=False, get_2d=False):
    """Gets YouMakeup images."""
    path = self.img_info[str(idx)][4] #3L-C3lsV2_w_6813_7819/3L-C3lsV2_w_7819.pt
    img = []
    if os.path.exists(self.img_tensor_path):
        img_path = self.img_tensor_path + path
        img = torch.load(img_path)
    else:
        print self.img_tensor_path+" does not exist!"
    return img
    
