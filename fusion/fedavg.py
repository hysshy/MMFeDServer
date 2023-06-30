import copy
from utils.Log import logger
import torch

def fedAvg(cfg_list, share_keys=['backbone']):
    #联邦平均融合算法
    net_w_lst = []
    for i in range(len(cfg_list)):
        client_epoch_file = cfg_list[i]['client_epoch_savepath']
        if  client_epoch_file != '':
            logger.info('加载节点模型:' + client_epoch_file)
            client_epoch_dict = torch.load(client_epoch_file, map_location='cpu')
            net_w_lst.append(client_epoch_dict['state_dict'])
    #融合全局基础模型backbone
    merge_w = copy.deepcopy(net_w_lst[0])
    sample_ratio = torch.tensor(1/len(net_w_lst), dtype=torch.float)
    for key in merge_w.keys():
        if key.split('.')[0] in share_keys:
            merge_w[key] = merge_w[key] * sample_ratio
    keylist = list(merge_w.keys())
    for key in keylist:
        start_key = key.split('.')[0]
        if start_key in share_keys:
            for i in range(1, len(net_w_lst)):
                merge_w[key] += net_w_lst[i][key] * sample_ratio

    #融合多任务下游任务模型

        elif start_key in copy_keys.keys():
            for i in range(len(cfg_list)):
                if cfg_list[i]['client_cfg'].data.train.ann_file.split('/')[-1].split('_')[0] == copy_keys[start_key]:
                    merge_w[key] = net_w_lst[i][key]
                    break
        else:
            del merge_w[key]
    del net_w_lst
    return merge_w

if __name__ == '__main__':
    net_w_lst = []
    client_epoch_dict1 = torch.load('/home/chase/shy/mutiltask_mmdetection/tools/work_dirs/faster_rcnn_r50_fpn_2x_coco_spjc/epoch_12.pth', map_location='cpu')
    net_w_lst.append(client_epoch_dict1['state_dict'])
    client_epoch_dict2 = torch.load('/home/chase/PycharmProjects/MMFeDServer/job/1/1_10.10.5.136/epoch_1.pth', map_location='cpu')
    net_w_lst.append(client_epoch_dict2['state_dict'])
    client_epoch_dict3 = torch.load('/home/chase/PycharmProjects/MMFeDServer/job/1/2_10.10.7.201/epoch_1.pth', map_location='cpu')
    net_w_lst.append(client_epoch_dict3['state_dict'])
    client_epoch_dict4 = torch.load('/home/chase/PycharmProjects/MMFeDServer/job/1/3_10.10.7.201/epoch_1.pth', map_location='cpu')
    net_w_lst.append(client_epoch_dict4['state_dict'])

    merge_w = copy.deepcopy(net_w_lst[0])
    share_keys = ['backbone', 'neck', 'rpn_head']
    sample_ratio = torch.tensor(1/len(net_w_lst), dtype=torch.float)
    for key in merge_w.keys():
        if key.split('.')[0] in share_keys:
            merge_w[key] = merge_w[key] * sample_ratio
    keylist = list(merge_w.keys())
    for key in keylist:
        for i in range(1, len(net_w_lst)):
            if key.split('.')[0] in share_keys:
                merge_w[key] += net_w_lst[i][key] * sample_ratio
            else:
                if key in merge_w.keys():
                    del merge_w[key]
    del net_w_lst