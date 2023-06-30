import copy
# from utils.Log import logger
import torch
import numpy as np

def fedHFGA(cfg_list, share_keys=['backbone', 'backbone_neck', 'attention_backbone']):
    #联邦平均融合算法
    net_w_dict = {}
    net_len = 0
    epoch_list = []
    for i in range(len(cfg_list)):
        client_epoch_file = cfg_list[i]['client_epoch_savepath']
        if  client_epoch_file != '':
            # logger.info('加载节点模型:' + client_epoch_file)
            client_epoch_dict = torch.load(client_epoch_file, map_location='cpu')
            epoch_list.append(client_epoch_dict['state_dict'])
            tasktype = cfg_list[i]['tasktype']
            if tasktype not in net_w_dict.keys():
                net_w_dict.setdefault(tasktype, [client_epoch_dict['state_dict']])
            else:
                net_w_dict[tasktype].append(client_epoch_dict['state_dict'])
            net_len += 1

    # 任务内部之间进行融合
    task_merge_w_dict = {}
    for tasktype, net_w in net_w_dict.items():
        task_merge_w = copy.deepcopy(net_w[0])
        sample_ratio = torch.tensor(1 / len(net_w), dtype=torch.float)
        for key in task_merge_w.keys():
            task_merge_w[key] = task_merge_w[key] * sample_ratio
            for i in range(1, len(net_w)):
                task_merge_w[key] += net_w[i][key] * sample_ratio
        task_merge_w_dict.setdefault(tasktype, task_merge_w)

    # 任务外部之间进行融合
    tasktypeList = list(task_merge_w_dict.keys())
    merge_w = task_merge_w_dict[tasktypeList[0]]
    sample_ratio = torch.tensor(1 / len(tasktypeList), dtype=torch.float)
    for key in merge_w.keys():
        if key.split('.')[0] in share_keys:
            merge_w[key] = merge_w[key] * sample_ratio
            for i in range(1, len(tasktypeList)):
                merge_w[key] += task_merge_w_dict[tasktypeList[i]][key] * sample_ratio
        else:
            task = key.split('.')[0].split('_')[-1]
            assert task in task_merge_w_dict.keys()
            merge_w[key] = task_merge_w_dict[task][key]

    # dist_list = []
    # for i in range(len(epoch_list)):
    #     dist = 0
    #     for key in merge_w.keys():
    #         if key.startswith('backbone.'):
    #             if key.endswith('num_batches_tracked'):
    #                 continue
    #             x = merge_w[key]
    #             y = epoch_list[i][key]
    #             dist = dist+torch.dist(x, y).numpy()
    #     dist_list.append(dist)
    # print(dist_list)
    # for i in range(len(cfg_list)):
    #     cfg_list[i]['adaptive_w'].append(dist_list[i]/np.sum(dist_list))
    #     print(cfg_list[i]['adaptive_w'])
    del task_merge_w_dict, net_w_dict
    return merge_w

if __name__ == '__main__':
    # dd = torch.load('/home/chase/shy/mutiltask_mmdetection/tools/work_dirs/faster_rcnn_r50_fpn_2x_coco_spjc2/epoch_1.pth')
    cfg_list = []
    cfg1 = {}
    cfg1['client_epoch_savepath'] = '/home/chase/PycharmProjects/MMFeDServer/job/Public-FedAvg_facecar/0_10.10.6.121/epoch_1.pth'
    cfg1['tasktype'] = 'faceDetect'
    cfg1['adaptive_w'] = 0
    cfg_list.append(cfg1)

    cfg2 = {}
    cfg2['client_epoch_savepath'] = '/home/chase/PycharmProjects/MMFeDServer/job/Public-FedAvg_facecar/1_10.10.6.121/epoch_1.pth'
    cfg2['tasktype'] = 'carplateDetect'
    cfg2['adaptive_w'] = 0
    cfg_list.append(cfg2)

    # cfg3 = {}
    # cfg3['client_epoch_savepath'] = '/home/chase/PycharmProjects/MMFeDServer/job/test-FPN2/2_172.16.1.190/epoch_1.pth'
    # cfg3['tasktype'] = 'faceDetect'
    # cfg3['adaptive_w'] = 0
    # cfg_list.append(cfg3)
    #
    # cfg4 = {}
    # cfg4['client_epoch_savepath'] = '/home/chase/PycharmProjects/MMFeDServer/job/test-FPN2/3_172.16.1.190/epoch_1.pth'
    # cfg4['tasktype'] = 'faceGender'
    # cfg4['adaptive_w'] = 0
    # cfg_list.append(cfg4)
    #
    # cfg5 = {}
    # cfg5['client_epoch_savepath'] = '/home/chase/PycharmProjects/MMFeDServer/job/test-FPN2/4_172.16.1.190/epoch_1.pth'
    # cfg5['tasktype'] = 'carplateDetect'
    # cfg5['adaptive_w'] = 0
    # cfg_list.append(cfg5)

    fedHFGA(cfg_list)