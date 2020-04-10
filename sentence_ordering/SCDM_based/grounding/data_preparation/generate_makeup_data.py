import numpy as np
import os, json, h5py, math, pdb, glob
import cv2
import shutil
import unicodedata
import pickle as pkl
import random
import string

np.set_printoptions(threshold=np.inf)
options = {}
options['feature_map_len']=[256,128,64,32,16]
options['scale_ratios_anchor1']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor2']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor3']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor4']=[0.25,0.5,0.75,1]
options['scale_ratios_anchor5']=[0.25,0.5,0.75,1]
options['pos_threshold'] = 0.5


SAMPLE_lEN = 1024
BATCH_SIZE = 4 
facial_info ={}


output_path = '../../data/h5py/'
train_id_path= '../../../../../YouMakeup/data/train/train_id'
dev_id_path= '../../../../../YouMakeup/data/valid/valid_id'
test_id_path = '../../../../../YouMakeup/data/task/step_ordering/test/test_id'
train_captions_path = '../../../../../YouMakeup/data/train/train_steps.json'
dev_captions_path = '../../../../../YouMakeup/data/valid/valid_steps.json'
dev_query_path = '../../../../../YouMakeup/data/task/step_ordering/valid/step_ordering_validation.json'
test_query_path = '../../../../../YouMakeup/data/task/step_ordering/test/step_ordering_test.json'
facial_dict_path = '../../data/facial_dict.json'
facial_map_path = '../../data/facial_map'

    
def json_read(json_path):
    file = open(json_path, 'r')
    items = []
    for line in file.readlines():
        dic = json.loads(line)
        items.append(dic)
    return items
    


def calculate_IOU(groundtruth, predict):
    groundtruth_init = max(0,groundtruth[0])
    groundtruth_end = groundtruth[1]
    predict_init = max(0,predict[0])
    predict_end = predict[1]
    init_min = min(groundtruth_init,predict_init)
    end_max = max(groundtruth_end,predict_end)
    init_max = max(groundtruth_init,predict_init)
    end_min = min(groundtruth_end,predict_end)
    if end_min < init_max:
        return 0
    IOU = ( end_min - init_max ) * 1.0 / ( end_max - init_min)
    return IOU
def calculate_facial_IOU(groundtruth, anchor):
    groundtruth_init = max(0,groundtruth[0])
    groundtruth_end = groundtruth[1]
    anchor_init = max(0,anchor[0])
    anchor_end = anchor[1]
    init_max = max(groundtruth_init,anchor_init)
    end_min = min(groundtruth_end,anchor_end)
    if end_min < init_max:
        return 0
    IOU = ( end_min - init_max ) * 1.0 / ( anchor_end - anchor_init)
    return IOU

train_j = json_read(train_captions_path)
dev_query_j = json_read(dev_query_path)
dev_j = json_read(dev_captions_path)
test_j = json_read(test_query_path)

def generate_facial_dict():
    facial_dict = {}
    info = train_j
    index = 0
    for item in info:
        steps = item['step']
        for idx,step in steps.items():
            
            zone_list = step['area']
            for zone in zone_list:
                if not zone in facial_dict.keys():
                    facial_dict[zone] = index
                    index += 1
    json_str = json.dumps(facial_dict, indent = 4)
    with open(facial_dict_path, 'w') as json_file:
        json_file.write(json_str)
    return 

def generate_anchor(feat_len,feat_ratio,max_len,output_path): # for 64 as an example
    anchor_list = []
    element_span = max_len / feat_len # 1024/64 = 16
    span_list = []
    for kk in feat_ratio:
        span_list.append(kk * element_span)
    for i in range(feat_len): # 64
        inner_list = []
        for span in span_list:
            left =   i*element_span + (element_span * 1 / 2 - span / 2)
            right =  i*element_span + (element_span * 1 / 2 + span / 2) 
            inner_list.append([left,right])
        anchor_list.append(inner_list)
    f = open(output_path,'w')
    f.write(str(anchor_list))
    f.close()
    return anchor_list


def generate_all_anchor():
    all_anchor_list = []
    for i in range(len(options['feature_map_len'])):
        anchor_list = generate_anchor(options['feature_map_len'][i],options['scale_ratios_anchor'+str(i+1)],SAMPLE_lEN,str(i+1)+'.txt')
        all_anchor_list.append(anchor_list)
    return all_anchor_list



def get_anchor_params_unit(anchor,ground_time_step):
    ground_check = ground_time_step[1]-ground_time_step[0]
    if ground_check <= 0:
        return [0.0,0.0,0.0]
    iou = calculate_IOU(ground_time_step,anchor)
    ground_len = ground_time_step[1]-ground_time_step[0]
    ground_center = (ground_time_step[1] - ground_time_step[0]) * 0.5 + ground_time_step[0]
    output_list  = [iou,ground_center,ground_len]
    return output_list


def generate_anchor_params(all_anchor_list,g_position):
    gt_output = np.zeros([len(options['feature_map_len']),max(options['feature_map_len']),len(options['scale_ratios_anchor1'])*3])
    for i in range(len(options['feature_map_len'])):
        for j in range(options['feature_map_len'][i]): 
            for k in range(len(options['scale_ratios_anchor1'])):
                input_anchor = all_anchor_list[i][j][k]
                output_temp = get_anchor_params_unit(input_anchor,g_position)
                gt_output[i,j,3*k:3*(k+1)]=np.array(output_temp)
    return gt_output

def get_facial_map_unit(facial_map,anchor, ground_time_step, facial_label):
    
    ground_check = ground_time_step[1] - ground_time_step[0]
    if ground_check <= 0:
        return facial_map
    iou = calculate_facial_IOU(ground_time_step,anchor)
    
    if iou > options['pos_threshold']:
        for item in facial_label:
            if not item in facial_info.keys():
                facial_info[item] = 0
            facial_info[item] += 1
            facial_map[facial_dict[item]] = 1
        
    return facial_map

def generate_facial_map_params(all_facial_map,all_anchor_list, g_position, facial_label):  
    
    #facial_len = len(facial_dict)
    #all_facial_map = np.zeros((len(options['feature_map_len']),max(options['feature_map_len']),len(options['scale_ratios_anchor1'])*facial_len))#(5,256,4*24)
    for i in range(len(options['feature_map_len'])):
        for j in range(options['feature_map_len'][i]): 
            for k in range(len(options['scale_ratios_anchor1'])):
                input_anchor = all_anchor_list[i][j][k]
                output_temp = get_facial_map_unit(all_facial_map[i,j,len(facial_dict)*k:len(facial_dict)*(k+1)],input_anchor,g_position, facial_label)
                all_facial_map[i,j,len(facial_dict)*k:len(facial_dict)*(k+1)]=np.array(output_temp)
    
    
    return all_facial_map
    


def get_ground_time(time):
    time=time.split(':')
    if len(time[0])!=2 or len(time[1])!=2 or len(time[2])!=2:
        print(time)
    return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])

def get_ground_truth_position(ground_position):
    left_frames = ground_position[0]
    right_frames = ground_position[1]
    left_position = int(left_frames / 29.4)
    right_position = int(right_frames / 29.4)
    if left_position < 0 or right_position < left_position:
        return -1,-1
    else:
        return left_position,right_position


def getlist(output_path, split):
    List = glob.glob(output_path+'/'+split+'/'+'*.h5')
    f = open(output_path+'/'+split+'/'+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')



def get_video_data_HL(video_data_path):
    files = open(video_data_path)
    List = []
    for ele in files:
        List.append(ele[:-1])
    return np.array(List)

def generate_facial_map():
    info = train_j + dev_j
    List = [line.rstrip('\n') for line in open(train_id_path,'r')] +  [line.rstrip('\n') for line in open(dev_id_path,'r')]
    
    if not os.path.exists(facial_map_path):
        os.makedirs(facial_map_path)
    all_anchor_list = generate_all_anchor()
    for item in info:
        video_name = item['video_id']
        
        if video_name not in List:
            #print(video_name,".mp4 is not in ",dataset," list")
            continue
        if os.path.exists(os.path.join(facial_map_path, video_name + '.npy')):
            print('feature map for %s has existed...'%(video_name))
            continue
        steps = item['step']   
        facial_map =  np.zeros((len(options['feature_map_len']),max(options['feature_map_len']),len(options['scale_ratios_anchor1']*len(facial_dict))))
        for idx,step in steps.items():
            facial_label = step['area']
            g_start = get_ground_time(step['startime'])
            g_end = get_ground_time(step['endtime'])
            g_time = [g_start, g_end]
                
            facial_map= generate_facial_map_params(facial_map,all_anchor_list,g_time,facial_label)#(5,256,4*24)

        np.save(os.path.join(facial_map_path,video_name),facial_map)

    
        

def driver(dataset, output_path):
    if dataset == 'train':
        info = train_j
        List = [line.rstrip('\n') for line in open(train_id_path,'r')]
        print('train_list_len:',len(List))
    elif dataset == 'dev_query':
        info = dev_query_j
        List = [line.rstrip('\n') for line in open(dev_id_path,'r')]
        print('dev_list_len:',len(List))
    elif dataset == 'dev':
        info = dev_j
        List = [line.rstrip('\n') for line in open(dev_id_path,'r')]
        print('dev_list_len:',len(List))
    elif dataset == 'test':
        info = test_j
        List = [line.rstrip('\n') for line in open(test_id_path,'r')]
        print('test_list_len:',len(List))
    if not os.path.exists(output_path+dataset):
        os.makedirs(output_path+dataset)

    all_anchor_list = generate_all_anchor()#(5,256,4,2)
    
    video_names_list = []
    video_rate_list = []
    sentence_list = [] 
    facial_label_list = []
    
    cnt = 0
    batch_id = 1
    if dataset == 'train' or dataset == 'dev':
        anchor_input_list= []
        ground_interval_list = []
        for item in info:
            video_name = item['video_id']
            
            if video_name not in List:
                print(video_name,".mp4 is not in ",dataset," list")
                continue
            steps = item['step']
                                        
            for idx,step in steps.items():
                facial_label = step['area']
                g_start = get_ground_time(step['startime'])
                g_end = get_ground_time(step['endtime'])
                if g_start > g_end or g_start < 0 or g_end < 0:
                    print(video_name,idx,"start_time or end time have wrong")
                    continue
                g_time = [g_start, g_end]
                anchor_input = generate_anchor_params(all_anchor_list,g_time)#(5,256,12)
                               
                #facial_map = generate_facial_map_params(all_anchor_list,g_time,facial_label)
                
                facial_label_array = np.zeros(len(facial_dict))
                for item in facial_label:
                    facial_label_array[facial_dict[item]] = 1
              
                video_names_list.append(str(video_name))

                sentence_list.append(unicodedata.normalize('NFKD', step['caption']).encode('ascii','ignore'))
                ground_interval_list.append(g_time)
                anchor_input_list.append(anchor_input)
                facial_label_list.append(facial_label_array)
                
                cnt+=1

                if cnt == BATCH_SIZE:
                    '''
                    print('video_name_list:',video_names_list)
                    print('video_rate_list:',video_rate_list)
                    print('sentence_list:',sentence_list)
                    print('ground_interval_list:',ground_interval_list)
                    print('anchor_input_list:',anchor_input_list)
                    '''
                    
                    batch = h5py.File(output_path+'/'+dataset+'/'+dataset+'_'+str(batch_id)+'.h5','w')
                    batch['video_name'] = np.string_(video_names_list) # batch_size
                    batch['sentence'] = np.string_(sentence_list) # batch_size   
                    batch['ground_interval'] = np.array(ground_interval_list) # batch_size x 2
                    batch['anchor_input'] = np.array(anchor_input_list)
                    batch['facial_label'] = np.array(facial_label_list)
                    
                    cnt = 0
                    batch_id += 1
                    video_names_list = []
                    sentence_list = []
                    ground_interval_list = []
                    anchor_input_list = []
                    facial_label_list = []
                    
                    
    elif dataset == 'dev_query' or dataset =='test':
        question_id_list = []
        ground_truth_list = []
        candidate_answer_list = []
        sentence_order_list = []
        querry_id = 1
        for item in info:
            video_name = item['video_id']
            if video_name not in List:
                #print(video_name,".mp4 is not in ",dataset," list")
                continue
            
            question_id = item['question_id']
            
            steps = item['step_caption']    
            
            candidate_answer = item['candidate_answer']
            for idx,step in steps.items():  
                video_names_list.append(str(video_name))
             
                sentence_list.append(unicodedata.normalize('NFKD', step).encode('ascii','ignore'))
                question_id_list.append(question_id)
                
                candidate_answer_list.append(candidate_answer)
                sentence_order_list.append(int(idx))
                cnt+=1

                if cnt == BATCH_SIZE:
                    batch = h5py.File(output_path+'/'+dataset+'/'+dataset+'_'+str(batch_id)+'.h5','w')
                    batch['video_name'] = np.string_(video_names_list) # batch_size
                    batch['sentence'] = np.string_(sentence_list) # batch_size
                    batch['candidate_answer'] = np.array(candidate_answer_list)
                    batch['question_id'] = np.array(question_id_list)
                    batch['sentence_order'] = np.array(sentence_order_list)
                    
                    cnt = 0
                    batch_id += 1
                    video_names_list = []
                    sentence_list = []
                    anchor_input_list = []
                    question_id_list = []
                    ground_truth_list = []
                    candidate_answer_list = []
                    sentence_order_list = []       
        

def shuffle_train_data():

    video_data_path_train = output_path+'train/train.txt'
    new_path = output_path+'shuffle_train/'
    if not os.path.exists(new_path):
        os.makedirs(new_path)


    video_list_train = get_video_data_HL(video_data_path_train)
    h5py_part_list = [[] for i in range(100)]
    for i in range(len(video_list_train)):
        index = i % 100
        h5py_part_list[index].append(video_list_train[i])

    count = 1
    for part_list in h5py_part_list:
        fname = []
        title = []
        anchor_input = np.zeros([BATCH_SIZE*len(part_list),5,256,12])+0.0
        timestamps = []
        
        facial_label = []
        
        
        for idx, item in enumerate(part_list):
            print(item)
            current_batch = h5py.File(item,'r')
            current_fname = current_batch['video_name']
            current_title = current_batch['sentence']
            current_timestamps = current_batch['ground_interval']
            
            current_anchor_input = current_batch['anchor_input']
            current_facial_label = current_batch['facial_label']
            
            
            fname = fname + list(current_fname)
            title = title + list(current_title)
            timestamps = timestamps + list(current_timestamps)
            
            facial_label = facial_label + list(current_facial_label)
            anchor_input[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,:,:,:] = current_anchor_input
        index = np.arange(BATCH_SIZE*len(part_list))
        np.random.shuffle(index)
        fname = [fname[i] for i in index]
        title =  [title[i] for i in index]
        timestamps = [timestamps[i] for i in index]
        
        anchor_input = anchor_input[index,:,:,:]
        facial_label = [facial_label[i] for i in index]
        
        for idx,item in enumerate(part_list):
            batch = h5py.File(new_path+'train'+str(count)+'.h5','w')
            batch['video_name'] = fname[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            batch['sentence'] = title[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            batch['ground_interval'] = timestamps[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            batch['anchor_input'] = anchor_input[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE,:,:,:]
            batch['facial_label'] =  facial_label[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            
            count = count + 1

    List = glob.glob(new_path+'*.h5')
    f = open(new_path+'train.txt','w')
    for ele in List:
        f.write(ele+'\n')


generate_facial_dict()

with open(facial_dict_path,'r') as load_f:     
    facial_dict = json.load(load_f)

generate_facial_map()

driver('train', output_path)
getlist(output_path,'train')

driver('dev', output_path)
getlist(output_path,'dev')

driver('dev_query', output_path)
getlist(output_path,'dev_query')

driver('test', output_path)
getlist(output_path,'test')


shuffle_train_data()

