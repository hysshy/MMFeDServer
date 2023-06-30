import torch

def adaptive(cfg_list):
    #联邦平均融合算法
    net_w_dict = {}
    net_len = 0
    for i in range(len(cfg_list)):
        client_epoch_file = cfg_list[i]['client_epoch_savepath']
        if  client_epoch_file != '':
            # logger.info('加载节点模型:' + client_epoch_file)
            client_epoch_dict = torch.load(client_epoch_file, map_location='cpu')
            tasktype = cfg_list[i]['tasktype']
            if tasktype not in net_w_dict.keys():
                net_w_dict.setdefault(tasktype, [client_epoch_dict['state_dict']])
            else:
                net_w_dict[tasktype].append(client_epoch_dict['state_dict'])
            net_len += 1