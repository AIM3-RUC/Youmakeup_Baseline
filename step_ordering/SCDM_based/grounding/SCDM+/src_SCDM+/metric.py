import numpy as np
import h5py
import json
import random
import logging
import Levenshtein
import operator
import math
from opt import *

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



def analysis_iou(result, epoch, logging):

    threshold_list = [0.1,0.3,0.5,0.7]
    rank_list = [1,5,10]
    result_dict = {}
    top1_iou = []

    for i in range(len(result)):
        print('\r',i,'/',len(result),end='     ')
        video_name = result[i][0]
        ground_truth_interval = result[i][1]
        sentence = result[i][2]
        predict_list = result[i][3]
        predict_score_list = result[i][4]

        iou_list = []
        for predict_interval in predict_list:
            iou_list.append(calculate_IOU(ground_truth_interval,predict_interval))
        top1_iou.append(iou_list[0])

        for rank in rank_list:
            for threshold in threshold_list:
                key_str = 'Recall@'+str(rank)+'_iou@'+str(threshold)
                if key_str not in result_dict:
                    result_dict[key_str] = 0

                for jj in range(rank):
                    if iou_list[jj] >= threshold:
                        result_dict[key_str] += 1
                        break

    logging.info('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    logging.info('epoch '+str(epoch)+': ')
    for key_str in result_dict:
        logging.info(key_str+': '+str(result_dict[key_str]*1.0/len(result)))

    logging.info('mean iou: '+str(np.mean(top1_iou)))
    logging.info('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    
    
def fuzzy_matching(predict_answer,candidate_answer):
    str_predict_answer = str(predict_answer)
    match_distance=[]
    for item in candidate_answer:       
        str_candidate_answer =str(list(item))
        distance = Levenshtein.distance(str_predict_answer,str_candidate_answer)
        
        match_distance.append(distance)
    close_answer = candidate_answer[match_distance.index(min(match_distance))]
   
    return close_answer
    
def output_multiple_prediction(result, epoch, answer_path = None):   
    result=sorted(result,key = lambda x:int(x[0]))
    prediction_dict = {}
    for idx in range(math.floor(len(result)/5)):
        question_id = result[idx*5][0]
        if not result[idx*5][0] == result[idx*5+1][0] == result[idx*5+2][0] == result[idx*5+3][0] == result[idx*5+4][0]:
            print("question_id has wrong")
            continue
        candidate_answer = result[idx*5][4]
       
        centers=[]
        orders=[]
        for i in range(5):
            sentence_order = result[idx*5+i][1]
            orders.append(sentence_order)
            predict_segment_list = result[idx*5+i][6]
            predict_score_list = result[idx*5+i][5]            
            centers.append((predict_segment_list[0][0]+predict_segment_list[0][1])/2) 

        if len(centers) != len(orders):
            print("something wrong with the tset results")
            continue
              
        predict_answer = [x for _,x in sorted(zip(centers,orders))] 
        predict_answer = fuzzy_matching(predict_answer,candidate_answer)
        
        prediction_dict[int(question_id)] = [ int(x) for x in predict_answer]       
    
         
    if answer_path:
        json_str = json.dumps(prediction_dict)
        with open(os.path.join(answer_path, str(epoch) +'.json'), 'w') as json_file:
            json_file.write(json_str)
  
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    print('epoch ' + str(epoch) + ': ')
    print('test result has generated successfully')
   
    print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

    
'''
if __name__ == '__main__':
    result_path = '/data3/rld/makeup/SCDM/grounding/makeup/result/scdm/29.npy' 
    result = np.load(result_path,allow_pickle=True)
    output_multiple_prediction(result,29, '/data3/rld/makeup/SCDM/grounding/makeup/result/scdm/test')
'''
