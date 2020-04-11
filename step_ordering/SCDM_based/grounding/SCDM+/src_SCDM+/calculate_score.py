import json,os
import numpy as np


def json_read(json_path):
    file = open(json_path, 'r')
    items = []
    for line in file.readlines():
        dic = json.loads(line)
        items.append(dic)
    return items
    
    
def calcualte_score(answer_json, truth_json):
    with open(answer_json, 'r') as f:
        answer = json.load(fp = f)
    truth = json_read(truth_json)
    if len(truth) != len(answer.keys()):
        print(answer_json,'does not contain all answers for questions ')
        exit(0)
    correct_num = 0
    for item in truth:
        question_id = item['question_id']
        groundtruth = item['groundtruth']
        if groundtruth == answer[str(question_id)]:
            correct_num +=1

    print('calculate accuracy for ', answer_json,':',correct_num/len(truth)*1.0)

if __name__ =='__main__':
    answer_json = '../result/dev/12.json'
    truth_json = '../../../../../../YouMakeup/data/task/step_ordering/valid/step_ordering_validation.json' 
    calcualte_score(answer_json,truth_json)