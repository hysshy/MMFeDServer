import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import time
import torch
from utils.Log import logger
from fedmerge.fedHFGA import fedHFGA
from fedmerge.fed_bl import fed_BL
from server.check_client import download_file, upload_file, get_client_loss, post_client_fedbl
from server import test_client
from utils.common import download_epoch_list, is_file_transfer_complete
import numpy as np
# ch = torch.load('/home/chase/PycharmProjects/MMFedClient/job/job0/epoch_1.pth', map_location='cpu')

def load_client_epoch_thread(aggregate_num, client_cfg):
    client_file_path = client_cfg['job_root'] + '/' + client_cfg['job_id'] + '/' + client_cfg[
        'client_id'] + '/epoch_' + str(aggregate_num) + '.pth'
    client_epoch_savepath = client_file_path
    while 1:
        status, info = download_file(client_file_path, client_epoch_savepath, client_cfg['client_ip'],
                                     client_cfg['client_port'])
        if status:
            client_cfg['client_epoch_savepath'] = client_epoch_savepath
            break
        else:
            logger.info(client_cfg['client_id']+':模型训练中,预计训练时间:'+info)
            time.sleep(float(info))


def load_client_loss_thread(client_cfg, bl_num, total_bl_num):
    client_work_dir= client_cfg['job_root'] + '/' + client_cfg['job_id'] + '/' + client_cfg[
        'client_id'] + '/'

    while 1:
        status, info = get_client_loss(client_cfg, client_work_dir, total_bl_num, bl_num)
        if status:
            client_cfg['loss'].append(float(info.split(':')[1]) / (client_cfg['bl_w'][-1]))
            break
        else:
            # logger.info(client_cfg['client_id']+':模型训练中,预计fed_bl_iter训练时间:'+info)
            time.sleep(float(info))

#获取节点loss
def load_client_loss(client_cfg_list, bl_num, total_bl_num):
    with ThreadPoolExecutor(max_workers=len(client_cfg_list)) as executor:
        futures = []
        for i in range(len(client_cfg_list)):
            client_cfg = client_cfg_list[i]
            futures.append(executor.submit(load_client_loss_thread, client_cfg, bl_num, total_bl_num))
        # 等待所有任务完成
        wait(futures)

#下载载节点模型
def load_client_epoch(client_cfg_list, aggregate_num):
    with ThreadPoolExecutor(max_workers=len(client_cfg_list)) as executor:
        futures = []
        for i in range(len(client_cfg_list)):
            client_cfg = client_cfg_list[i]
            futures.append(executor.submit(load_client_epoch_thread, aggregate_num, client_cfg))
        # 等待所有任务完成
        wait(futures)


#检查客户端模型是全部否发送完成
def check_client_epoch(client_cfg_list, aggregate_num):
    for i in range(len(client_cfg_list)):
        client_cfg = client_cfg_list[i]
        client_file_path = client_cfg['job_root'] + '/' + client_cfg['job_id'] + '/' + client_cfg[
            'client_id'] + '/epoch_' + str(aggregate_num) + '.pth'
        while 1:
            # if client_file_path in download_epoch_list:
            if os.path.exists(client_file_path):
                assert is_file_transfer_complete(client_file_path)
                client_cfg['client_epoch_savepath'] = client_file_path
                break
            time.sleep(5)


#发布融合模型
def publish_client(client_cfg_list, epochfile):
    published_ips = []
    for i in range(len(client_cfg_list)):
        client_cfg = client_cfg_list[i]
        if client_cfg['client_ip'] not in published_ips:
            client_savepath = client_cfg['job_root'] + '/' + client_cfg['job_id']
            _, status = upload_file(epochfile, client_savepath, client_cfg['client_ip'], client_cfg['client_port'])
            assert status
            published_ips.append(client_cfg['client_ip'])

#清空缓存
def clear_client(client_cfg_list, savefile):
    for i in range(len(client_cfg_list)):
        os.remove(client_cfg_list[i]['client_epoch_savepath'])
        client_cfg_list[i]['client_epoch_savepath'] = ''
        # 删除中间融合模型
        if os.path.exists(client_cfg_list[i]['merge_file']):
            os.remove(client_cfg_list[i]['merge_file'])
        client_cfg_list[i]['merge_file'] = savefile

def merge_epoch(client_cfg_list, aggregate_num, test_interval=1):
    logger.info('联邦融合epoch:' + str(aggregate_num))
    logger.info('加载所有节点模型')
    check_client_epoch(client_cfg_list, aggregate_num)
    # load_client_epoch(client_cfg_list, aggregate_num)
    logger.info('完成加载所有节点模型,开始融合')
    merge_w = fedHFGA(client_cfg_list)
    savefile = client_cfg_list[0]['job_root'] + '/' + client_cfg_list[0]['job_id'] + '/merge_epoch_' + str(
        aggregate_num) + '.pth'
    torch.save(merge_w, savefile)
    logger.info('完成融合，清楚节点缓存')
    clear_client(client_cfg_list, savefile)
    logger.info('向客户端发布融合模型')
    publish_client(client_cfg_list, savefile)
    if aggregate_num % test_interval == 0:
        logger.info('测试模型准确率')
        test_client.eval(client_cfg_list)

# 多任务loss均衡
def fed_bl_client(client_cfg_list, total_bl_num):
    for fedbl_num in range(total_bl_num):
        load_client_loss(client_cfg_list, fedbl_num+1, total_bl_num)
        fed_BL(fedbl_num+1, client_cfg_list)
        for i in range(len(client_cfg_list)):
            client_cfg = client_cfg_list[i]
            client_work_dir = client_cfg['job_root'] + '/' + client_cfg['job_id'] + '/' + client_cfg[
                'client_id'] + '/'
            status = post_client_fedbl(client_cfg, client_work_dir, fedbl_num+1, client_cfg['bl_w'][-1])

#融合节点模型
def fed_merging(client_cfg_list, fed_start_from, max_epochs=24, test_interval=1, total_bl_num=1, fedbl=False):
    #联邦融合
    for aggregate_num in range(fed_start_from, max_epochs+1):
        # fed_bl_client(client_cfg_list, total_bl_num)
        if fedbl:
            t = threading.Thread(target=fed_bl_client, args=(client_cfg_list, total_bl_num))
            t.start()
        merge_epoch(client_cfg_list, aggregate_num, test_interval)

