from server.merge_client import merge_epoch
from utils.Log import logger

def resume_train(cfg, client_config_list):
    fed_start_from = 1
    if cfg.resume_from is not None:
        logger.info('resume training')
        aggregate_num = int(cfg.resume_from.split('_')[-1].split('.')[0])
        merge_epoch(client_config_list, aggregate_num)
        fed_start_from += aggregate_num
    return fed_start_from