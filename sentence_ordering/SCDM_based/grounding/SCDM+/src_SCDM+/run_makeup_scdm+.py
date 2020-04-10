#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import pdb
import time
import json
from collections import defaultdict
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import rnn_cell
from keras.preprocessing import sequence
import unicodedata
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
import logging
import string
from model import SSAD_SCDM
import random
import pickle as pkl
from utils import *
import math
import operator
from metric import *
from opt import *



finetune = False
PRETRAINED_EPOCH = 17

options = default_options()
all_video_fts = h5py.File(options['video_fts_path'],'r')


optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer,
                    "adam":tf.train.AdamOptimizer,
                    "gradientdescent":tf.train.GradientDescentOptimizer,
                    "adagrad":tf.train.AdagradOptimizer}


def make_prepare_path(task):
    
    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    log_file_name = 'screen_output_'+task+'_'+str(time_stamp)+'.log'
    fh = logging.FileHandler(filename=log_file_name, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    logging.root.addHandler(fh)

    model_save_dir = options['model_save_dir'] 
    result_save_dir = options['result_save_dir'] 
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(os.path.join(result_save_dir , task)):
        os.makedirs(os.path.join(result_save_dir , task))
    if not os.path.exists(options['words_path']):
        os.makedirs(options['words_path'])

    return logging, model_save_dir, result_save_dir



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract a CNN features')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default = 1, type=int)
    parser.add_argument('--net', dest='model',
                        help='model to test',
                        default = None, type=str)
    parser.add_argument('--task', dest='task',
                        help='train or test',
                        default='train', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick
    union = list(map(operator.sub, x2, x1)) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)
        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick

def generate_anchor(feat_len,feat_ratio,max_len): # for 64 as an example
    anchor_list = []
    element_span = max_len / feat_len # 1024/64 = 16
    span_list = []
    for kk in feat_ratio:
        span_list.append(kk * element_span)
    for i in range(feat_len): # 64
        inner_list = []
        for span in span_list:
            left = i*element_span + (element_span * 1 / 2 - span / 2)
            right = i*element_span + (element_span * 1 / 2 + span / 2) 
            inner_list.append([left,right])
        anchor_list.append(inner_list)
    return anchor_list



def generate_all_anchor():
    all_anchor_list = []
    for i in range(len(options['feature_map_len'])):
        anchor_list = generate_anchor(options['feature_map_len'][i],options['scale_ratios_anchor'+str(i+1)],options['sample_len'])
        all_anchor_list.append(anchor_list)
    return all_anchor_list



def get_word_embedding(word_embedding_path, wordtoix_path, ixtoword_path, extracted_word_fts_init_path):
    print('loading word features ...')
    word_fts_dict = np.load(word_embedding_path,encoding = 'bytes',allow_pickle = True).tolist()
    wordtoix = np.load(wordtoix_path, encoding = 'bytes', allow_pickle = True).tolist()
    ixtoword = np.load(ixtoword_path, encoding = 'bytes', allow_pickle = True).tolist()
    word_num = len(wordtoix)
    extract_word_fts = np.random.uniform(-3,3,[word_num,300]) 
    count = 0
    for index in range(word_num):
        if ixtoword[index] in word_fts_dict:
            extract_word_fts[index] = word_fts_dict[ ixtoword[index] ]
            count = count + 1
    np.save(extracted_word_fts_init_path,extract_word_fts)



def preProBuildWordVocab(logging,sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    logging.info('preprocessing word counts and creating vocab based on word count threshold {:d}'.format(word_count_threshold))
    word_counts = {} # count the word number
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1  # if w is not in word_counts, will insert {w:0} into the dict

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    logging.info('filtered words from {:d} to {:d}'.format(len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


def get_video_data_HL(video_data_path):
    files = open(video_data_path)
    List = []
    for ele in files:
        List.append(os.path.join('../',ele[:-1]))
    return np.array(List)


def get_video_data_jukin(video_data_path_train,video_data_path_dev, video_data_path_test):
    title = []
    video_list_train = get_video_data_HL(video_data_path_train) # get h5 file list
    video_list_dev = get_video_data_HL(video_data_path_dev)
    video_list_test = get_video_data_HL(video_data_path_test)
    train_sentence_num = 0
    dev_sentence_num = 0
    test_sentence_num = 0
    for ele in video_list_train:
        print(ele)
        batch_data = h5py.File(ele,'r')
        batch_fname = batch_data['video_name']
        batch_title = batch_data['sentence']
        train_sentence_num += len(batch_title)
        for i in range(len(batch_fname)):
            title.append(bytes.decode(batch_title[i]))
    for ele in video_list_dev:
        print(ele)
        batch_data = h5py.File(ele,'r')
        batch_fname = batch_data['video_name']
        batch_title = batch_data['sentence']
        dev_sentence_num += len(batch_title)
        for i in range(len(batch_fname)):
            title.append(bytes.decode(batch_title[i]))
    for ele in video_list_test:
        print(ele)
        batch_data = h5py.File(ele,'r')
        batch_fname = batch_data['video_name']
        batch_title = batch_data['sentence']
        test_sentence_num += len(batch_title)
        for i in range(len(batch_fname)):
            title.append(bytes.decode(batch_title[i]))
    print('Train sentence num:',train_sentence_num)
    print('Dev sentence num:',dev_sentence_num)
    print('Test sentence num:',test_sentence_num)
    title = np.array(title)
    video_caption_data = pd.DataFrame({'Description':title})
    return video_caption_data, video_list_train, video_list_dev, video_list_test


def generate_video_fts_data(video_name):
    #video_feat_path = os.path.join(options['video_fts_path'],bytes.decode(video_name))    
    #video_fts = np.load(video_feat_path,allow_pickle=True)
    
    video_fts = all_video_fts[bytes.decode(video_name)]['i3d_rgb_features']
    video_fts_shape = np.shape(video_fts)
    video_clip_num = video_fts_shape[0]
    video_fts_dim = video_fts_shape[1]
    output_video_fts = np.zeros([1,options['sample_len'],video_fts_dim])+0.0
    for i in range(video_clip_num):
        output_video_fts[0,i,:] = video_fts[i,:]
        if i == options['sample_len']-1:
            return output_video_fts, video_clip_num
    return output_video_fts, video_clip_num


def generate_batch_video_fts(current_video_name):
    video_fts_list = []
    video_duration_list = []
    for video_name in current_video_name:
        video_fts,video_duration = generate_video_fts_data(video_name)
        video_fts_list.append(video_fts)
        video_duration_list.append(video_duration)
    return np.array(video_fts_list), np.array(video_duration_list)

def generate_batch_facial_map(current_video_name):
    facial_map_list = []
    for video_name in current_video_name:
        facial_map_path = os.path.join(options['facial_map_path'],bytes.decode(video_name)+'.npy')
        facial_map_list.append(np.load(facial_map_path,allow_pickle=True)+0.0)
        
    return np.array(facial_map_list)


def train(logging, model_save_dir, result_save_dir):

    if not os.path.exists(options['word_fts_path']):
        meta_data, train_data, dev_data, test_data = get_video_data_jukin(options['video_data_path_train'],options['video_data_path_dev'] ,options['video_data_path_test'])
        captions = meta_data['Description'].values
        for c in string.punctuation:
            captions = map(lambda x: x.replace(c, ''), captions)
        wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(logging, captions, word_count_threshold=1)
        np.save(options['ixtoword_path'], ixtoword)
        np.save(options['wordtoix_path'], wordtoix)
        get_word_embedding(options['word_embedding_path'],options['wordtoix_path'],options['ixtoword_path'],options['word_fts_path'])
        word_emb_init = np.array(np.load(options['word_fts_path'], encoding = 'bytes', allow_pickle = True).tolist(),np.float32)
    else:
        wordtoix = (np.load(options['wordtoix_path'],allow_pickle=True)).tolist()
        ixtoword = (np.load(options['ixtoword_path'],allow_pickle=True)).tolist()
        word_emb_init = np.array(np.load(options['word_fts_path'], encoding = 'bytes', allow_pickle = True).tolist(),np.float32)
        train_data = get_video_data_HL(options['video_data_path_train']) # get h5 file list

    if finetune:
        start_epoch = PRETRAINED_EPOCH 
        MODEL = model_save_dir+'/model-'+str(start_epoch-1)


    model = SSAD_SCDM(options,word_emb_init)
    inputs, outputs = model.build_train()
    t_loss = outputs['loss_all']
    t_loss_ssad = outputs['loss_ssad'] 
    t_loss_regular = outputs['reg_loss']
    t_positive_loss_all = outputs['positive_loss_all']
    t_hard_negative_loss_all = outputs['hard_negative_loss_all']
    t_easy_negative_loss_all = outputs['easy_negative_loss_all'] 
    t_smooth_center_loss_all = outputs['smooth_center_loss_all']
    t_smooth_width_loss_all = outputs['smooth_width_loss_all']
    t_facial_positive_loss_all = outputs['facial_positive_loss_all']
    t_facial_negative_loss_all = outputs['facial_negative_loss_all']
    t_facial_possitive_num = outputs['facial_possitive_num']
    t_facial_possitive_loss = outputs['facial_possitive_loss']
    t_facial_negative_num = outputs['facial_negative_num']
    t_facial_negative_loss = outputs['facial_negative_loss']
    t_predict_overlap = outputs['predict_overlap']
    t_predict_reg = outputs['predict_reg']
    t_predict_label_map = outputs['predict_facial_map']
    t_facial_map_accuracy_dict = outputs['facial_map_true'] 
    t_facial_map_precision_dict = outputs['facial_map_true_true'] 
    #print('t_facial_negative_loss_all.type:',type(t_facial_negative_loss_all))

    t_feature_segment = inputs['feature_segment']
    t_sentence_index_placeholder = inputs['sentence_index_placeholder']
    t_sentence_w_len = inputs['sentence_w_len']
    t_gt_overlap = inputs['gt_overlap']
    t_gt_facial_map = inputs['gt_facial_map']
    
    
    

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu90% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically 
    sess = tf.InteractiveSession(config=config)

    optimizer = optimizer_factory[options['optimizer']](**options['opt_arg'][options['optimizer']])
    if options['clip']:
        gvs = optimizer.compute_gradients(t_loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
    else:
        train_op = optimizer.minimize(t_loss)

    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep=400)
    tf.initialize_all_variables().run()

    with tf.device("/cpu:0"):
        if finetune:
            saver.restore(sess, MODEL)
    ############################################# start training ####################################################
    loss_all_dict ={'mean_loss':[],'mean_ssad':[],'mean_positive_loss':[],'mean_hard_negative_loss':[],'mean_easy_negative_loss':[],'mean_smooth_center_loss':[],'mean_smooth_width_loss':[],'mean_facial_positive_loss':[],'mean_facial_negative_loss':[]}
    tStart_total = time.time()
   
    for epoch in range(options['max_epochs']):
        facial_map_accuracy_dict_all = {}
        facial_map_precision_dict_all = {}
        facial_map_gt_dict = {}
        for key in options['facial_dict'].keys():
            facial_map_accuracy_dict_all[key] = 0.0
            facial_map_precision_dict_all[key] = 0.0
            facial_map_gt_dict[key] = 0.0

        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]

        tStart_epoch = time.time()

        loss_list = np.zeros(len(train_data)) # each item in loss_epoch record the loss of this h5 file
        loss_ssad_list = np.zeros(len(train_data))
        loss_positive_loss_all_list = np.zeros(len(train_data))
        loss_hard_negative_loss_all_list = np.zeros(len(train_data))
        loss_easy_negative_loss_all_list = np.zeros(len(train_data))
        loss_smooth_center_loss_all_list = np.zeros(len(train_data))
        loss_smooth_width_loss_all_list = np.zeros(len(train_data))
        loss_facial_positive_loss_all_list = np.zeros(len(train_data))
        loss_facial_negative_loss_all_list = np.zeros(len(train_data))

        for current_batch_file_idx in range(len(train_data)):
            print('\r ',current_batch_file_idx,'/',len(train_data),end = '    ')
            #logging.info("current_batch_file_idx = {:d}".format(current_batch_file_idx))
            #logging.info(train_data[current_batch_file_idx])

            tStart = time.time()
            current_batch = h5py.File(train_data[current_batch_file_idx],'r')

            # processing sentence
            current_captions_tmp = current_batch['sentence']
            current_captions = []
            for ind in range(options['batch_size']):
                current_captions.append(bytes.decode(current_captions_tmp[ind]))
            current_captions = np.array(current_captions)
            for ind in range(options['batch_size']):
                for c in string.punctuation: 
                    current_captions[ind] = current_captions[ind].replace(c,'')
            for i in range(options['batch_size']):
                current_captions[i] = current_captions[i].strip()
                if current_captions[i] == '':
                    current_captions[i] = '.'
            current_caption_ind = list(map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions))
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=options['max_sen_len'] -1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_length = np.array( list(map(lambda x: (x != 0).sum(), current_caption_matrix ))) # save the sentence length of this batch

            # processing video
            current_video_name = current_batch['video_name']            
            
            current_facial_map = generate_batch_facial_map(current_video_name)
        
            
            current_anchor_input = np.array(current_batch['anchor_input'])            
            #print('***********************************************current_anchor_input.shape',np.shape(current_anchor_input))
            current_video_feats, current_video_duration =  generate_batch_video_fts(current_video_name)
            
            _,  loss, loss_ssad, positive_loss_all, hard_negative_loss_all, easy_negative_loss_all,\
                smooth_center_loss_all, smooth_width_loss_all, loss_regular, facial_positive_loss_all, \
                facial_negative_loss_all,possitive_loss, possitive_num, negative_loss, negative_num, facial_map_accuracy_dict, facial_map_precision_dict,predict_label_map = sess.run(
                    [train_op, t_loss, t_loss_ssad , t_positive_loss_all, t_hard_negative_loss_all, \
                     t_easy_negative_loss_all, t_smooth_center_loss_all, t_smooth_width_loss_all, t_loss_regular,\
                     t_facial_positive_loss_all, t_facial_negative_loss_all,t_facial_possitive_loss,t_facial_possitive_num,\
                     t_facial_negative_loss,t_facial_negative_num,t_facial_map_accuracy_dict,t_facial_map_precision_dict,t_predict_label_map], \
                    feed_dict={
                        t_feature_segment: current_video_feats,
                        t_sentence_index_placeholder: current_caption_matrix,
                        t_sentence_w_len: current_caption_length,
                        t_gt_overlap: current_anchor_input,
                        t_gt_facial_map: current_facial_map
                        })
            for key in options['facial_dict'].keys():
                facial_map_accuracy_dict_all[key] = facial_map_accuracy_dict_all[key] + facial_map_accuracy_dict[key]
                facial_map_precision_dict_all[key] = facial_map_precision_dict_all[key]+ facial_map_precision_dict[key]
                for i in range(len(options['feature_map_len'])):
                    facial_map_gt_dict[key] += np.sum(current_facial_map[:, i:i+1, :options['feature_map_len'][i], int(options['facial_dict'][key]):len(options['scale_ratios_anchor%d'%(i+1)])*len(options['facial_dict']):len(options['facial_dict'])])
            
            loss_list[current_batch_file_idx] = loss
            loss_ssad_list[current_batch_file_idx] = loss_ssad
            loss_positive_loss_all_list[current_batch_file_idx] = positive_loss_all
            loss_hard_negative_loss_all_list[current_batch_file_idx] = hard_negative_loss_all
            loss_easy_negative_loss_all_list[current_batch_file_idx] = easy_negative_loss_all
            loss_smooth_center_loss_all_list[current_batch_file_idx] = smooth_center_loss_all
            loss_smooth_width_loss_all_list[current_batch_file_idx] = smooth_width_loss_all
            loss_facial_positive_loss_all_list[current_batch_file_idx] = facial_positive_loss_all
            loss_facial_negative_loss_all_list[current_batch_file_idx] = facial_negative_loss_all
            
            #logging.info("loss = {:f} loss_ssad = {:f} loss_regular = {:f} positive_loss_all = {:f} hard_negative_loss_all = {:f} easy_negative_loss_all = {:f} smooth_center_loss_all = {:f} smooth_width_loss_all = {:f}".format(loss, loss_ssad, loss_regular, positive_loss_all, hard_negative_loss_all, easy_negative_loss_all, smooth_center_loss_all, smooth_width_loss_all))

        if finetune:
            logging.info("Epoch: {:d} done.".format(epoch+start_epoch))
        else:
            logging.info("Epoch: {:d} done.".format(epoch))
        tStop_epoch = time.time()
        logging.info('Epoch Time Cost: {:f} s'.format(round(tStop_epoch - tStart_epoch,2)))

        logging.info('Current Epoch Mean loss {:f}'.format(np.mean(loss_list)))
        logging.info('Current Epoch Mean loss_ssad {:f}'.format(np.mean(loss_ssad_list)))
        logging.info('Current Epoch Mean positive_loss_all {:f}'.format(np.mean(loss_positive_loss_all_list)))
        logging.info('Current Epoch Mean hard_negative_loss_all {:f}'.format(np.mean(loss_hard_negative_loss_all_list)))
        logging.info('Current Epoch Mean easy_negative_loss_all {:f}'.format(np.mean(loss_easy_negative_loss_all_list)))
        logging.info('Current Epoch Mean smooth_center_loss_all {:f}'.format(np.mean(loss_smooth_center_loss_all_list)))
        logging.info('Current Epoch Mean smooth_width_loss_all {:f}'.format(np.mean(loss_smooth_width_loss_all_list)))
        logging.info('Current Epoch Mean facial_positive_loss_all {:f}'.format(np.mean(loss_facial_positive_loss_all_list)))
        logging.info('Current Epoch Mean negative_negative_loss_all {:f}'.format(np.mean(loss_facial_negative_loss_all_list)))
        for key in options['facial_dict'].keys():
            logging.info('Current Epoch classifier-{:s} accuracy-{:.4f}  precision-{:.4f}  GT-{:f}'.format(key, facial_map_accuracy_dict_all[key] / (len(train_data) * options['batch_size'] * sum(options['feature_map_len'])*len(options['scale_ratios_anchor1'])), facial_map_precision_dict_all[key] / max( 0.1, facial_map_gt_dict[key]), facial_map_gt_dict[key]))

        loss_all_dict['mean_loss'].append(np.mean(loss_list))
        loss_all_dict['mean_ssad'].append(np.mean(loss_ssad_list))
        loss_all_dict['mean_positive_loss'].append(np.mean(loss_positive_loss_all_list))
        loss_all_dict['mean_hard_negative_loss'].append(np.mean(loss_hard_negative_loss_all_list))
        loss_all_dict['mean_easy_negative_loss'].append(np.mean(loss_easy_negative_loss_all_list))
        loss_all_dict['mean_smooth_center_loss'].append(np.mean(loss_smooth_center_loss_all_list))
        loss_all_dict['mean_smooth_width_loss'].append(np.mean(loss_smooth_width_loss_all_list))
        loss_all_dict['mean_facial_positive_loss'].append(np.mean(loss_facial_positive_loss_all_list))
        loss_all_dict['mean_facial_negative_loss'].append(np.mean(loss_facial_negative_loss_all_list))
        #################################################### test ################################################################################################
        if finetune:
            now_epoch = epoch + start_epoch
        else:
            now_epoch = epoch
            
        if np.mod(epoch, 1) == 0 and epoch > 4:
            if finetune:
                logging.info('Epoch {:d} is done. Saving the model ...'.format(now_epoch))
            else:
                logging.info('Epoch {:d} is done. Saving the model ...'.format(now_epoch))
                
            with tf.device("/cpu:0"):
                if finetune:
                    saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=now_epoch)
                else:
                    saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=now_epoch)
            

    logging.info("Finally, saving the model ...")
    with tf.device("/cpu:0"):
        if finetune:
            saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=now_epoch)
        else:
            saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=now_epoch)
          
    np.save(os.path.join(options['result_save_dir'] + task,'epoch_loss.npy'), loss_all_dict)
    tStop_total = time.time()
    logging.info("Total Time Cost: {:f} s".format(round(tStop_total - tStart_total,2)))

def dev(model_save_dir, result_save_dir, task):
    all_anchor_list = generate_all_anchor()
    wordtoix = (np.load(options['wordtoix_path'],allow_pickle = True)).tolist()
    ixtoword = (np.load(options['ixtoword_path'],allow_pickle = True)).tolist()
    word_emb_init = np.array(np.load(options['word_fts_path'], encoding = 'bytes', allow_pickle=True).tolist(),np.float32)
    model = SSAD_SCDM(options,word_emb_init)
    
    test_data = get_video_data_HL(options['video_data_path_dev'])
    inputs,t_predict_overlap,t_predict_reg, t_facial_map_accuracy_dict, t_facial_map_precision_dict = model.build_dev_inference()
    t_feature_segment = inputs['feature_segment']
    t_sentence_index_placeholder = inputs['sentence_index_placeholder']
    t_sentence_w_len = inputs['sentence_w_len']
    t_facial_map = inputs['gt_facial_map']

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically
    sess = tf.InteractiveSession(config = config)

    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep = 200)
        latest_checkpoint = tf.train.latest_checkpoint(model_save_dir)
        tmp = latest_checkpoint.split('model-')
        all_epoch = int(tmp[1])
            
    epoch_list = range(all_epoch + 1)
    
    test_epoch_start = 29
    for epoch in epoch_list:
        epoch_exact = epoch + test_epoch_start
        facial_map_accuracy_dict_all = {}
        facial_map_precision_dict_all = {}
        facial_map_gt_dict = {}

        for  facial_key in options['facial_dict'].keys():
            facial_map_accuracy_dict_all[facial_key] = 0
            facial_map_precision_dict_all[facial_key] = 0
            facial_map_gt_dict[facial_key] = 0

        if os.path.exists(result_save_dir+'/'+ task +'/' +str(epoch_exact)+'.npy'):
            print(result_save_dir,'/',task ,'/' ,str(epoch_exact),'.npy has existed')
            continue

        with tf.device("/cpu:0"):
            saver.restore(sess, options['model_save_dir'] + 'model-' + str(epoch_exact))

        result = []
        for current_batch_file_idx in range(len(test_data)):
            print('\r ', current_batch_file_idx, '/', len(test_data), end = '      ')
            current_batch = h5py.File(test_data[current_batch_file_idx],'r')
            # processing sentence
            current_captions_tmp = current_batch['sentence']
            current_captions = []
            for ind in range(options['batch_size']):
                current_captions.append(bytes.decode(current_captions_tmp[ind]))
            current_captions = np.array(current_captions)
            for ind in range(options['batch_size']):
                for c in string.punctuation:
                    current_captions[ind] = current_captions[ind].replace(c,'')
            for i in range(options['batch_size']):
                current_captions[i] = current_captions[i].strip()
                if current_captions[i] == '':
                    current_captions[i] = '.'
            current_caption_ind = list(map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions))
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=options['max_sen_len']-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_length = np.array( list(map(lambda x: (x != 0).sum(), current_caption_matrix ))) # save the sentence length of this batch

            # processing video
            current_video_name = current_batch['video_name']
            current_video_feats, current_video_duration =  generate_batch_video_fts(current_video_name)

            current_facial_map = generate_batch_facial_map(current_video_name)
            predict_overlap, predict_reg, facial_map_accuracy_dict, facial_map_precision_dict = sess.run(
                [t_predict_overlap, t_predict_reg, t_facial_map_accuracy_dict, t_facial_map_precision_dict],
                feed_dict = {
                    t_feature_segment: current_video_feats,
                    t_sentence_index_placeholder: current_caption_matrix,
                    t_sentence_w_len: current_caption_length,
                    t_facial_map: current_facial_map
                    })
            for key in options['facial_dict'].keys():
                facial_map_accuracy_dict_all[key] = facial_map_accuracy_dict_all[key] + facial_map_accuracy_dict[key]
                facial_map_precision_dict_all[key] = facial_map_precision_dict_all[key]+ facial_map_precision_dict[key]
                for i in range(len(options['feature_map_len'])):
                    facial_map_gt_dict[key] += np.sum(current_facial_map[:, i:i+1, :options['feature_map_len'][i], int(options['facial_dict'][key]):len(options['scale_ratios_anchor%d'%(i+1)])*len(options['facial_dict']):len(options['facial_dict'])])

            for batch_id in range(options['batch_size']):
                predict_overlap_list = []
                predict_center_list = []
                predict_width_list = []
                expand_anchor_list = []
                for anchor_group_id in range(len(options['feature_map_len'])):
                    for anchor_id in range(options['feature_map_len'][anchor_group_id]):
                        for kk in range(4):
                            current_anchor = all_anchor_list[anchor_group_id][anchor_id][kk]
                            current_len = current_anchor[1] - current_anchor[0]
                            if current_len >= 1.0 and current_len <= 64.0:
                                predict_overlap_list.append(predict_overlap[anchor_group_id][batch_id,0,anchor_id,kk])
                                predict_center_list.append(predict_reg[anchor_group_id][batch_id,0,anchor_id,kk*2])
                                predict_width_list.append(predict_reg[anchor_group_id][batch_id,0,anchor_id,kk*2+1])
                                expand_anchor_list.append(all_anchor_list[anchor_group_id][anchor_id][kk])

                a_left = []
                a_right = []
                a_score = []
                for index in range(len(predict_overlap_list)):
                    anchor = expand_anchor_list[index]
                    anchor_center = (anchor[1] - anchor[0]) * 0.5 + anchor[0]
                    anchor_width = anchor[1] - anchor[0]
                    center_offset = predict_center_list[index]
                    width_offset = predict_width_list[index]
                    p_center = anchor_center + 0.1 * anchor_width * center_offset
                    p_width = anchor_width * np.exp(0.1 * width_offset)
                    p_left = max(0, p_center - p_width * 0.5)  # frame
                    p_right = min(options['sample_len'], p_center + p_width * 0.5) # frame
                    if p_right - p_left > current_video_duration[batch_id]:
                        continue
                    a_left.append(p_left)
                    a_right.append(p_right)
                    a_score.append(predict_overlap_list[index])
                picks = nms_temporal(a_left,a_right,a_score,0.5)
                process_segment = []
                process_score = []
                for pick in picks:
                    process_segment.append([a_left[pick],a_right[pick]])
                    process_score.append(a_score[pick])
                
                result.append([current_batch['video_name'][batch_id],\
                               current_batch['ground_interval'][batch_id],\
                               current_batch['sentence'][batch_id],\
                               process_segment,\
                               process_score])


        np.save(result_save_dir+'/'+ task + '/' +str(epoch_exact)+'.npy',result)
        logging.info('***************************************************************')

        analysis_iou(result,epoch_exact,logging)
        for key in options['facial_dict'].keys():
            logging.info('Current Epoch classifier-{:s} accuracy-{:.4f}  precision-{:.4f}  GT-{:f}'.format(key, facial_map_accuracy_dict_all[key] / (len(test_data) * options['batch_size'] * options['sample_len']), facial_map_precision_dict_all[key] / max( 0.1, facial_map_gt_dict[key]), facial_map_gt_dict[key]))

    logging.info('***************************************************************')                                

def test(model_save_dir, result_save_dir, task):

    all_anchor_list = generate_all_anchor()

    wordtoix = (np.load(options['wordtoix_path'],allow_pickle = True)).tolist()
    ixtoword = (np.load(options['ixtoword_path'],allow_pickle = True)).tolist()
    word_emb_init = np.array(np.load(options['word_fts_path'], encoding = 'bytes', allow_pickle = True).tolist(),np.float32)
    model = SSAD_SCDM(options, word_emb_init)
     
    test_data = get_video_data_HL(options['video_data_path_test'])
    inputs,t_predict_overlap,t_predict_reg = model.build_test_inference()
    t_feature_segment = inputs['feature_segment']
    t_sentence_index_placeholder = inputs['sentence_index_placeholder']
    t_sentence_w_len = inputs['sentence_w_len']
       

    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.3 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically 
    sess = tf.InteractiveSession(config = config)

    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep = 200)
        latest_checkpoint = tf.train.latest_checkpoint(model_save_dir)

        tmp = latest_checkpoint.split('model-')
        all_epoch = int(tmp[1])
        
    epoch_list = range(all_epoch + 1)
          

    test_epoch_start = 25
    for epoch in epoch_list:        
        epoch_exact = epoch + test_epoch_start
        if os.path.exists(result_save_dir+'/'+ task +'/' +str(epoch_exact)+'.npy'):
            print(result_save_dir,'/',task ,'/' ,str(epoch_exact),'.npy has existed')
            continue

        with tf.device("/cpu:0"):
            saver.restore(sess, tmp[0] + 'model-' + str(epoch_exact))

        result = []
        for current_batch_file_idx in range(len(test_data)):

            print('\r ',current_batch_file_idx,'/',len(test_data),end = '      ')
            
            current_batch = h5py.File(test_data[current_batch_file_idx],'r')
            # processing sentence
            current_captions_tmp = current_batch['sentence']
            current_captions = []
            for ind in range(options['batch_size']):
                current_captions.append(bytes.decode(current_captions_tmp[ind]))
            current_captions = np.array(current_captions)
            for ind in range(options['batch_size']):
                for c in string.punctuation: 
                    current_captions[ind] = current_captions[ind].replace(c,'')
            for i in range(options['batch_size']):
                current_captions[i] = current_captions[i].strip()
                if current_captions[i] == '':
                    current_captions[i] = '.'
            current_caption_ind = list(map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions))
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=options['max_sen_len']-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_length = np.array( list(map(lambda x: (x != 0).sum(), current_caption_matrix ))) # save the sentence length of this batch

            # processing video
            current_video_name = current_batch['video_name']
            current_video_feats, current_video_duration =  generate_batch_video_fts(current_video_name)
            
            predict_overlap, predict_reg = sess.run(
                [t_predict_overlap, t_predict_reg],
                feed_dict = {
                    t_feature_segment: current_video_feats,
                    t_sentence_index_placeholder: current_caption_matrix,
                    t_sentence_w_len: current_caption_length,
                        })
                        
            for batch_id in range(options['batch_size']):
                predict_overlap_list = []
                predict_center_list = []
                predict_width_list = []
                expand_anchor_list = []
                for anchor_group_id in range(len(options['feature_map_len'])):
                    for anchor_id in range(options['feature_map_len'][anchor_group_id]):
                        for kk in range(4):
                            current_anchor = all_anchor_list[anchor_group_id][anchor_id][kk]
                            current_len = current_anchor[1] - current_anchor[0] 
                            if current_len >= 1.0 and current_len <= 64.0:
                                predict_overlap_list.append(predict_overlap[anchor_group_id][batch_id,0,anchor_id,kk])
                                predict_center_list.append(predict_reg[anchor_group_id][batch_id,0,anchor_id,kk*2])
                                predict_width_list.append(predict_reg[anchor_group_id][batch_id,0,anchor_id,kk*2+1])
                                expand_anchor_list.append(all_anchor_list[anchor_group_id][anchor_id][kk])

                a_left = []
                a_right = []
                a_score = []
                for index in range(len(predict_overlap_list)):
                    anchor = expand_anchor_list[index]
                    anchor_center = (anchor[1] - anchor[0]) * 0.5 + anchor[0]
                    anchor_width = anchor[1] - anchor[0]
                    center_offset = predict_center_list[index]
                    width_offset = predict_width_list[index]
                    p_center = anchor_center + 0.1 * anchor_width * center_offset
                    p_width = anchor_width * np.exp(0.1 * width_offset)
                    p_left = max(0, p_center-p_width * 0.5)  # frame
                    p_right = min(options['sample_len'], p_center + p_width * 0.5) # frame
                    if p_right - p_left > current_video_duration[batch_id]:
                        continue
                    a_left.append(p_left)
                    a_right.append(p_right)
                    a_score.append(predict_overlap_list[index])
                picks = nms_temporal(a_left,a_right,a_score,0.5)
                process_segment = []
                process_score = []
                for pick in picks:
                    process_segment.append([a_left[pick],a_right[pick]])                    
                    process_score.append(a_score[pick])
                    
                result.append([int(current_batch['question_id'][batch_id]),\
                               current_batch['sentence_order'][batch_id],\
                               current_batch['video_name'][batch_id],\
                               current_batch['sentence'][batch_id],\
                               current_batch['candidate_answer'][batch_id],\
                               process_score,\
                               process_segment])
                    
        np.save(result_save_dir+'/'+ task + '/' + str(epoch_exact) +'.npy',result)
        logging.info('***************************************************************')
        output_multiple_prediction(result,epoch_exact,logging)
        logging.info('***************************************************************')




if __name__ == '__main__':
    
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    logging, model_save_dir, result_save_dir = make_prepare_path(args.task)
    if args.task == 'train': 
        train( logging, model_save_dir, result_save_dir)
    if args.task == 'dev': 
        dev(model_save_dir, result_save_dir, args.task)
    if args.task == 'test':
        test(model_save_dir, result_save_dir, args.task)
    
