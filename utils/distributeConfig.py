import os
from mmcv import Config, ConfigDict

import copy

# 将 Config 转换为字典格式
def configTojson(config):
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for option in config.options(section):
            config_dict[section][option] = config.get(section, option)
    return config_dict

def init_config(config):
    client_cfg_List = []
    print(config)
    cfg = Config.fromfile(config)
    for i in range(len(cfg.data)):
        client_cfg = copy.deepcopy(cfg)
        client_ip = cfg.data[i].client_ip
        client_port = cfg.data[i].client_port
        tasktype = cfg.data[i].tasktype
        client_cfg.gpu_ids = cfg.data[i].gpu_ids
        client_cfg.data = cfg.data[i]
        client_id = '{}_{}'.format(str(i),str(client_ip))
        client_savepath = cfg.job_root+'/'+cfg.job_id+'/'+client_id
        if not os.path.exists(client_savepath):
            os.makedirs(client_savepath)
        client_cfg.dump(client_savepath+'/'+config.split('/')[-1])
        client_cfg_List.append({'client_ip':client_ip,
                                'client_port':client_port,
                                'client_id':client_id,
                                'client_cfg':client_cfg,
                                'tasktype':tasktype,
                                'client_epoch_savepath':'',
                                'job_root':client_cfg.job_root,
                                'job_id':client_cfg.job_id,
                                'epoch_num':0,
                                'merge_file':'',
                                'client_cfg_file':client_savepath+'/'+config.split('/')[-1],
                                'loss':[],
                                'bl_w':[1]})
    return client_cfg_List, cfg
