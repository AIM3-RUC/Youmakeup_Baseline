# -*- coding: utf-8 -*-
import makeup_test_retrieval
import datasets,main
import numpy as np
import torch
from tqdm import tqdm as tqdm
import json
import img_text_composition_models
import argparse

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--mod_data_path', type=str, default='./') 
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    opt = parse_opt()
    
    #get all texts
    f = open(opt.mod_data_path+'mod_data.json','r')
    data = json.load(f)
    mods = data['train']['mods']
    texts = [mod['to_str'] for mod in mods]    


    #load tirg model checkpoint
    checkpoint_path = opt.checkpoint
    model = None
    checkpoint = torch.load(checkpoint_path)
    raw_opt = checkpoint['opt']
    model = img_text_composition_models.Concat(texts, embed_dim=raw_opt.embed_dim)
    model = model.cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    raw_opt.dataset_path = '../shared_data/'
    trainset, testset = main.load_dataset(raw_opt)
    makeup_test_retrieval.test(raw_opt, model, testset)