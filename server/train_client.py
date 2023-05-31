from utils.Log import logger
import requests

def init_trainer(cfg):
    logger.info('初始化训练模型 ：'+ cfg['client_ip'])
    reponse = requests.post('http://{}:{}/init_trainer'.format(cfg['client_ip'], cfg['client_port']),
                           data=cfg['client_cfg_file'])
    if reponse.text == 'success':
        return True
    else:
        return False

def start_train(cfg):
    logger.info('开始训练:' + cfg['client_id']+'-'+cfg['client_ip'])
    reponse = requests.post('http://{}:{}/start_train'.format(cfg['client_ip'], cfg['client_port']),
                           data=cfg['client_cfg_file'])
    if reponse.text == 'success':
        return True
    else:
        return False

def train(cfg):
    assert start_train(cfg)
