import os
import json
import numpy as np
import PIL
import skimage.io
import torch
import torch.utils.data
import torchvision
import warnings
import random
import itertools
import img_text_composition_models
import datasets
import main
import time
import random
from scipy.stats import pearsonr
from Levenshtein import *
import numpy as np
import copy
import argparse

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--mod_data_path', type=str, default='./') 
    parser.add_argument('--split', type=str, default='test') 
    args = parser.parse_args()
    return args

def Get_Validation_Info(json_path):
    qinfos = []
    with open(json_path, "r") as file:
        for index,f in enumerate(file):
            line  = json.loads(f)
            question_info = {}
            
            question_info['question_id'] = line['question_id']# number: 1 (related image dir name: 1/)
            question_info['video_id']  = line['video_id']#"-9GYpCvGIgM"
            question_info['step_caption'] = line['step_caption']#list of step captions
            question_info['groundtruth'] = line['groundtruth'] #[4, 1, 2, 5, 3]
            question_info['candidate_answer'] = line['candidate_answer']
            qinfos.append(question_info)
    return qinfos  

'''
input:
- an init img
- an init caption
output:
- a image order
'''
def image_retrieval(query,rest_imgs,img_feas):
    for i in range(query.shape[0]):
        query[i,:] /= np.linalg.norm(query[i,:])            
    cal_sim = query.dot(img_feas.T).reshape(5,) 
    for i in range(5):
        if i not in rest_imgs:
            cal_sim[i] = -1.0 
        
    score = max(cal_sim)
    img_index = cal_sim.argsort()[-1]
    
    return score,img_index
    
def get_next_image(init_img_id, init_caption_id, all_captions, all_imgs, rest_imgs, img_feas, model):
    source_img = [all_imgs[init_img_id]]
    source_img = torch.stack(source_img).float()
    source_img = torch.autograd.Variable(source_img).cuda()
    mod_text = [all_captions[init_caption_id+1]]
    init_query = model.compose_img_text(source_img,mod_text).data.cpu().numpy()
    score,img_index = image_retrieval(init_query, rest_imgs, img_feas)
    
    next_score = score+1 #init
    cur_cid = init_caption_id+1 #init
    get_img_id = 0  
    rest_cnum = len(all_captions) - cur_cid # rest captions num
    if len(rest_imgs) < rest_cnum:
        while len(rest_imgs) < rest_cnum:
            mod_text[0] = mod_text[0]+' '+all_captions[cur_cid+1]#update
            cur_cid += 1#update
            next_query = model.compose_img_text(source_img,mod_text).data.cpu().numpy()
            next_score,next_img_index = image_retrieval(next_query, rest_imgs, img_feas)
            rest_cnum = rest_cnum - 1#update
            if next_score < score:
                #get_img_id = img_index
                cur_cid -= 1
                break
            else:
                score = next_score
                img_index = next_img_index
        get_img_id =  img_index
    else:
        get_img_id = img_index
    return get_img_id, cur_cid, score

def get_img_order(init_pic,all_captions, all_imgs, img_feas, model):
    init_img_id = init_pic
    answer = []
    answer.append(init_img_id+1)
    rest_imgs = [0,1,2,3,4]
    rest_imgs.remove(init_img_id)
    #find init caption id
    init_caption_id = 0
    step_split = []
    score = 0.0
    for i in range(4):
        get_img_id, cur_cid,scorei = get_next_image(init_img_id, init_caption_id, all_captions, all_imgs, rest_imgs,img_feas,model)
        init_img_id = get_img_id
        init_caption_id = cur_cid
        step_split.append(cur_cid)
        #update rest imgs, remove target images
        rest_imgs.remove(get_img_id)
        answer.append(get_img_id+1)
        score += scorei
    return answer, score, step_split

def get_img_feature(imgs):
    imgs = torch.stack(imgs).float()
    imgs = torch.autograd.Variable(imgs).cuda()
    with torch.no_grad():
        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
    for i in range(imgs.shape[0]):
        imgs[i,:] /= np.linalg.norm(imgs[i,:]) 
    return imgs

def if_same(select_answer,ground_truth):
    count = 0
    for i in range(len(select_answer)):
        count += (select_answer[i] == ground_truth[i])
    return int(count == 5)

def Get_QueryImages(imgs_path):
    imgs = []
    transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])
    img_name = sorted(os.listdir(imgs_path))
    #print(img_name)
    for n in img_name:
        img_path = imgs_path + n
        #read image
        if img_path[-4:] != '.jpg':
            continue
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB') 
            img = transform(img)
            imgs.append(img)
    return imgs

def json_data_save(path, data):
    jsdata = json.dumps(data)
    jsfile = open(path, 'w')
    jsfile.write(jsdata)
    jsfile.close() 
    

if __name__ == '__main__':
    opt = parse_opt()
    save_split = opt.split
    
    img_path = {}
    img_path['valid'] = "../../../YouMakeup/data/task/image_ordering/valid/images/"
    img_path['test'] = "../../../YouMakeup/data/task/image_ordering/test/images/"
    images_path = img_path[save_split]
    
    question_path = {}
    question_path['valid'] = "../../../YouMakeup/data/task/image_ordering/valid/image_ordering_validation.json"
    question_path['test'] = "../../../YouMakeup/data/task/image_ordering/test/image_ordering_test.json"
    questions_info = Get_Validation_Info(question_path[save_split])
    #load init_pics
    init_pics_load = {}
    init_pics_load['valid'] = np.load('init_pics_dev_0.534.npy') #init image calculate by pair-wise model
    init_pics_load['test'] = np.load('init_pics_test_0.572.npy') #init image calculate by pair-wise model
    init_pics = init_pics_load[save_split]
    
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
    
    answer_save = {}
    correct_Levenshtein = 0
    choose_answer_Levenshtein = None
    for i, query in enumerate(questions_info):
        captions = query['step_caption']
        groundtruth = query['groundtruth']
        answers = query['candidate_answer'] 
        print "-----------------------------------------query {}---------------------------------------".format(i+1)
        #get five query imgs
        img_dir = str(query['question_id'])
        imgs_path = images_path + img_dir + '/'
        imgs = Get_QueryImages(imgs_path)
    
        img_features = get_img_feature(imgs)
        tic = time.time()
        predict_answer,score,step_split = get_img_order(init_pics[i]-1,captions, imgs, img_features, model)
        max_relevancy = -1
        for answer in query['candidate_answer']:
            s1 = [str(s) for s in predict_answer]
            s2 = [str(s) for s in answer]
            s1 = "".join((s1))
            s2 = "".join((s2))
            score = ratio(s1,s2)
            if score > max_relevancy:
                max_relevancy = score
                choose_answer_Levenshtein = answer
                 

        count_Levenshtein = if_same(choose_answer_Levenshtein, groundtruth)
        correct_Levenshtein += count_Levenshtein
        answer_save[query['question_id']] = choose_answer_Levenshtein
        print "ground truth:",groundtruth,"used time",time.time()-tic 
        print "choose_answer:",choose_answer_Levenshtein,bool(count_Levenshtein)
    acc = float(correct_Levenshtein)/len(questions_info)
    print "choose accuracy:",acc
        
    path = "tirg_{}_answer{:.3f}.json".format(save_split,acc)
    json_data_save(path, answer_save)
    print "The choose answers has been saved to {}".format(path)
    
    