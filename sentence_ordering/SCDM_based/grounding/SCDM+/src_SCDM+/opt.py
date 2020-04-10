"""
Default hyper parameters

Use this setting can obtain similar or even better performance as original SST paper
"""

from collections import OrderedDict
import numpy as np
import sys
import json
import time
import os

def default_options():

    options = OrderedDict()# use OrderedDict,the elements will be ordered by adding orders

    #*** MODEL CONFIG ***#
    options['single_feat_dim'] = 1024
    options['video_feat_dim'] = options['single_feat_dim'] # dim of video feature

    options['feature_map_len']=[256,128,64,32,16]
    options['scale_ratios_anchor1']=[0.25,0.5,0.75,1]
    options['scale_ratios_anchor2']=[0.25,0.5,0.75,1]
    options['scale_ratios_anchor3']=[0.25,0.5,0.75,1]
    options['scale_ratios_anchor4']=[0.25,0.5,0.75,1]
    options['scale_ratios_anchor5']=[0.25,0.5,0.75,1]

    options['reg_dim'] = 2        #center offset && width offset
    options['weight_anchor'] = [1,1,1,1,1]

    options['batch_size'] = 4      # training batch size

    options['learning_rate'] = 0.0001 # initial learning rate (I fix learning rate to 1e-3 during training phase)
    options['reg'] = 0.001           # regularization strength (control L2 regularization ratio)
    options['max_epochs'] = 40    # maximum training epochs to run
    options['sample_len'] = 1024      # the length ratio of the sampled stream compared to the video
                                      # the video was split to 1024 clips,every clip 1s
    options['pos_threshold'] = 0.5
    options['neg_threshold'] = 0.5
    
    options['facial_label_pos_threshold'] = 0.5
    options['facial_label_neg_threshold'] = 0.5
    options['hard_neg_threshold'] = 0.1

    options['posloss_weight'] = 100.0
    options['hardnegloss_weight'] = 50.0
    options['easynegloss_weight'] = 50.0
    options['reg_weight_center'] = 10.0
    options['reg_weight_width'] = 10.0
    options['facialposloss_weigth'] = 0.5
    options['facialnegloss_weigth'] = 0
    options['word_embedding_path'] ='../../../data/glove.840B.300d_dict.npy'
    options['max_sen_len'] = 20
    options['num_layers'] = 1
    options['dim_hidden'] = 256

    options['SRU'] = True
    options['bias'] = True
    options['dropout'] = 0.2
    options['zoneout'] = None
    options['facial_map_path'] = '../../../data/facial_map'
    options['video_fts_path'] = '../../../data/makeup_i3d_rgb_stride_1s.hdf5'
    #options['video_fts_path'] = '../../../data/makeup_c3d_rgb_stride_1s.hdf5'
    options['video_data_path_train'] = '../../../data/h5py/shuffle_train/train.txt'
    options['video_data_path_dev'] = '../../../data/h5py/dev/dev.txt'
    #options['video_data_path_test'] = '../../../data/h5py/dev_query/dev_query.txt'
    options['video_data_path_test'] = '../../../data/h5py/test/test.txt'
    options['wordtoix_path'] = '../words/wordtoix.npy'
    options['ixtoword_path'] = '../words/ixtoword.npy'
    options['word_fts_path'] = '../words/word_glove_fts_init.npy'
    options['model_save_dir'] = '../model/'
    options['result_save_dir'] = '../result/'
    options['words_path'] = '../words/'
    options['fical_dict_path'] = '../../../data/facial_dict.json'
    with open(options['fical_dict_path'],'r') as load_f:
        options['facial_dict'] = json.load(load_f) 
    options['optimizer'] = "adam" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    options['clip'] = True # clip gradient norm
    options['norm'] = 5.0 # global norm
    options['opt_arg'] = {'adam':{'learning_rate':options['learning_rate'], 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8}}
    
    return options

