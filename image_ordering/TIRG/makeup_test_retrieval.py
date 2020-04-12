#-*- coding:utf-8 -*-
"""Evaluates the retrieval model on YouMakeup dataset."""
import numpy as np
import torch
from tqdm import tqdm as tqdm


def test(opt, model, testset):
    """Tests a model over the given testset."""
    model.eval()
    
    test_queries_videos = testset.get_test_queries()
    vnum = len(test_queries_videos)
    
    out = {} # save R@1,2,3,5 value for each video
    for k in [1,2,3,5]:
        out[k] = 0.0
    for vid in tqdm(test_queries_videos):
        out[vid] = []
        test_queries = test_queries_videos[vid]
        to_img_ids = testset.to_img_ids[vid]
        target_imgs = testset.target_imgs[vid]
        all_target_imgs = []
        all_target_img_ids = []
        all_target_captions = []
        all_queries = []  #source img + mod caption = query, save all queries
        all_query_captions = [] #Store captions corresponding to all source imgs
        all_query_target_img_ids = [] #Store the groundtruth of the target_img_id of all queries
        if test_queries:
            # compute test query features
            imgs = []
            mods = []
            for t in test_queries:
                imgs += [testset.get_img(t['source_img_id'])] #source image
                mods += [t['mod']['str']] #step caption
                if len(imgs) >= opt.batch_size or t is test_queries[-1]: #calculate by batch_size
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float()
                    imgs = torch.autograd.Variable(imgs).cuda()
                    #imgs = torch.autograd.Variable(imgs)
                    #mods = [t.decode('utf-8') for t in mods]
                    with torch.no_grad():
                        f = model.compose_img_text(imgs, mods).data.cpu().numpy()
                    #f = model.compose_img_text(imgs, mods).data.detach().numpy()
                    all_queries += [f]
                    imgs = []
                    mods = []
            all_queries = np.concatenate(all_queries)
            all_query_captions = [t['target_caption'] for t in test_queries]
            all_query_target_img_ids = [t['target_img_id'] for t in test_queries]
    
            # compute all image features
            imgs = []
            for i in range(len(to_img_ids)): 
                imgs += [testset.get_img(to_img_ids[i])]
                if len(imgs) >= opt.batch_size or i == len(to_img_ids) - 1:
                    if 'torch' not in str(type(imgs[0])):
                        imgs = [torch.from_numpy(d).float() for d in imgs]
                    imgs = torch.stack(imgs).float()
                    imgs = torch.autograd.Variable(imgs).cuda()
                    #imgs = torch.autograd.Variable(imgs)
                    with torch.no_grad():
                        imgs = model.extract_img_feature(imgs).data.cpu().numpy()
                        #imgs = model.extract_img_feature(imgs).data.detach().numpy()
                    all_target_imgs += [imgs]
                    imgs = []
            all_target_imgs = np.concatenate(all_target_imgs)
            all_target_captions = [img['captions'][0] for img in target_imgs]
            all_target_img_ids = to_img_ids

        # feature normalization
        for i in range(all_queries.shape[0]):
            all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
        for i in range(all_target_imgs.shape[0]):
            all_target_imgs[i, :] /= np.linalg.norm(all_target_imgs[i, :])

        # match test queries to target images, get nearest neighbors
        sims = all_queries.dot(all_target_imgs.T)
  
        # remove query image in the target imgs set
        find_target_img_index = 0
        if test_queries:
            for i, t in enumerate(test_queries):
                source_img_id = t['source_img_id']
                for p in range(0,len(to_img_ids)):
                    if source_img_id == to_img_ids[p]:
                        sims[i, p] = -10e10  # remove query image
        
        nn_result = [np.argsort(-sims[i, :]) for i in range(sims.shape[0])] 

        # compute recalls    
        nn_result = np.array([[all_target_img_ids[nn] for nn in nns] for nns in nn_result])

        for k in [1,2,3,5]:
            r = 0.0
            s = 0.0
            for i, nns in enumerate(nn_result):
                s += float(k)/len(nns)
                if all_query_target_img_ids[i] in nns[:k]:
                    r += 1
                    #print(score[i][all_query_target_img_ids[i]])
            r /= len(nn_result)
            s /= len(nn_result)
            out[k] += r
 
    for k in [1,2,3,5]:
        out[k] /= vnum
    
    results = []
    for k in [1,2,3,5]:
        results += [('recall_top' + str(k) + '_correct_composition', out[k])]
        print 'recall_top' + str(k) + '_correct_composition', out[k]
    return results





