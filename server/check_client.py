import requests
from utils.Log import logger
import time
import threading
lock = threading.Lock()
#检查客户端是否在线
def is_client_online(cfg):
    logger.info('检查客户端是否在线:'+ cfg['client_ip'])
    status = True
    reponse = requests.post('http://{}:{}/check_online'.format(cfg['client_ip'], cfg['client_port']),
                           json={'mmdet': ''})
    if reponse.status_code != 200:
        status = False
    logger.info(status)
    return status

# 检查客户端环境
def is_envs_prepare(cfg):
    logger.info('检查客户端环境:'+ cfg['client_ip'])
    reponse = requests.post('http://{}:{}/check_envs'.format(cfg['client_ip'], cfg['client_port']),
                           json={'mmdet': ''})
    reponse = reponse.json()
    status = True
    for k,v in reponse.items():
        if v == '':
            status = False
        logger.info(k+':'+v)
    return status

#向客户端发送文件
def upload_file(filepath, savepath, ip, port):
    logger.info('向客户端发送文件-' + ip +':'+filepath)
    with open(filepath, 'rb') as f:
        response = requests.post('http://{}:{}/download_file'.format(ip, port), files={'file': f}, data={'savepath':savepath})
    # 检查响应状态码是否为 200
    if response.status_code == 200:
        logger.info('File uploaded successfully')
        return response, True
    else:
        logger.info('Failed to upload file')
        return response, False

#下载客户端文件
def download_file(client_file_path, file_savepath, ip, port):
    with lock:
        logger.info('下载客户端文件-' + ip +':'+client_file_path)
        response = requests.post('http://{}:{}/upload_file'.format(ip, port), data={'filepath':client_file_path})
        # 检查响应状态码是否为 200
        if response.status_code == 200:
            # 将响应内容写入本地文件
            logger.info('开始下载文件到:' + file_savepath)
            with open(file_savepath, 'wb') as f:
                f.write(response.content)
            logger.info('File download successfully')
            return True, file_savepath
        else:
            # logger.info('File download fail')
            epoch_time = response.text
            return False, epoch_time

#发送训练文件
def send_netpy(cfg):
        logger.info('向客户端发送训练文件:' + cfg['client_id']+'-'+cfg['client_ip'])
        savedir = cfg['job_root']+'/'+cfg['job_id']+'/'+cfg['client_id']
        response, status = upload_file(cfg['client_cfg_file'], savedir, cfg['client_ip'], cfg['client_port'])
        if status:
            cfg['client_cfg_file'] = response.text
        return status

#发送融合模型
def send_merge_epoch(cfg):
    with lock:
        logger.info('向客户端发送融合模型:' + cfg['client_id']+'-'+cfg['client_ip'])
        savedir = cfg['job_root']+'/'+cfg['job_id']
        response, status = upload_file(cfg['merge_file'], savedir, cfg['client_ip'], cfg['client_port'])
        return status

# 检查客户端数据集
def is_dataset_prepaer(cfg):
    logger.info('检查客户端数据集:'+ cfg['client_id']+'-'+cfg['client_ip'])
    reponse = requests.post('http://{}:{}/check_dataset'.format(cfg['client_ip'], cfg['client_port']),
                           data=cfg['client_cfg_file'])
    reponse = reponse.json()
    status = True
    for k, v in reponse.items():
        if not v:
            status = False
        logger.info(k+':'+str(v))
    return status

#检查客户端
def is_prepare(cfg):
    logger.info('检查客户端:'+ cfg['client_id']+'-'+cfg['client_ip'])
    if is_client_online(cfg) \
            and is_envs_prepare(cfg) \
            and send_netpy(cfg) \
            and is_dataset_prepaer(cfg):
        return True
    else:
        return False

#获取客户端当前loss值
def get_client_loss(cfg, work_dir, total_fedbl_num, fedbl_num):
    # logger.info('获取客户端当前loss值:'+ cfg['client_id'])
    response = requests.post('http://{}:{}/post_client_loss'.format(cfg['client_ip'], cfg['client_port']),
                           json={'work_dir': work_dir,
                                 'total_fedbl_num':total_fedbl_num,
                                 'fedbl_num':fedbl_num})
    if response.status_code == 200:
        logger.info('成功获取客户端当前loss值'+ cfg['client_id']+':'+response.text)
        return True, response.text
    else:
        return False, response.text

#向客户端发送adaptivew值
def post_client_fedbl(cfg, work_dir, fedbl_num, bl_w):
    logger.info('向客户端发送bl_w值:'+ cfg['client_id']+'-'+str(bl_w))
    response = requests.post('http://{}:{}/get_client_fedbl'.format(cfg['client_ip'], cfg['client_port']),
                           json={'work_dir': work_dir,
                                 'fedbl_num':fedbl_num,
                                 'bl_w':bl_w})
    return response.text

# #向客户端发送adaptive_w值
# def post_client_bl_w(cfg, work_dir, tasktype, bl_w):
#     logger.info('向客户端发送bl_w值:'+ cfg['client_id']+'-'+str(bl_w))
#     response = requests.post('http://{}:{}/get_client_bl_w'.format(cfg['client_ip'], cfg['client_port']),
#                            json={'work_dir': work_dir,
#                                  'tasktype':tasktype,
#                                  'bl_w':bl_w})
#     return response.text
# def get_epoch_thread(cfg):
#     client_epoch_num = 1
#     cfg_files = cfg['client_cfg_file'].split('/')
#     client_work_dir = ''
#     for i in range(len(cfg_files) - 1):
#         client_work_dir = client_work_dir + cfg_files[i] + '/'
#     while True:
#         logger.info('获取客户端模型:' + cfg['client_id'] + '-' + cfg['client_ip'])
#         response = requests.get('http://{}:{}/upload_epoch'.format(cfg['client_ip'], cfg['client_port']),
#                                 data=client_work_dir+'epoch_'+str(client_epoch_num)+'.pth')
#         # 检查响应状态码是否为 200
#         if response.status_code == 200:
#             # 将响应内容写入本地文件
#             logger.info('开始下载文件 :' + cfg['client_ip'])
#             # filename = response.headers['Content-Disposition'].split('=')[-1]
#             filename = 'epoch_'+str(client_epoch_num)+'.pth'
#             with open(cfg['client_root']+'/'+filename, 'wb') as f:
#                 f.write(response.content)
#             logger.info('File downloaded successfully')
#             if cfg['client_cfg'].max_epochs <= client_epoch_num:
#                 logger.info('等待客户端模型:' + cfg['client_id'] + '-' + cfg['client_ip'] + ' 训练完毕')
#                 break
#             cfg['epoch_num'] = client_epoch_num
#             client_epoch_num += 1
#         else:
#             logger.info('等待客户端模型:' + cfg['client_id']+'-'+cfg['client_ip'])
#             time.sleep(5)
#
# #下载客户端模型
# def download_epoch(cfg):
#     # get_epoch_thread(cfg)
#     t = threading.Thread(target=get_epoch_thread, args=(cfg,))
#     t.start()
#     # while(1):
#     #     time.sleep(10)
