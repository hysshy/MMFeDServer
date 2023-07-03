import argparse
import json
import threading
import time
from utils.distributeConfig import init_config
from server.to_client import is_prepare
from server.train_client import train
from server.merge_client import fed_merging
from server.resume_client import resume_train
from server import app
from flask_restful import Api

def parse_args():
    parser = argparse.ArgumentParser(description='MMFed Train')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('ip', help='front ip')
    parser.add_argument('port', help='front port')
    args = parser.parse_args()
    return args

def start(config, server_ip, sever_port):
    client_config_list, cfg = init_config(config)
    #检查节点是否在线
    for i in range(len(client_config_list)):
        assert is_prepare(client_config_list[i])
    #resume训练
    fed_start_from = resume_train(cfg, client_config_list)
    #开始训练
    for i in range(len(client_config_list)):
        train(client_config_list[i])
    #联邦融合
    # time.sleep(30)
    t = threading.Thread(target=fed_merging, args=(client_config_list, fed_start_from, cfg.max_epochs, cfg.test_interval, cfg.total_fedbl_num, cfg.fedbl))
    t.start()
    # fed_merging(client_config_list, 1, cfg.max_epochs, cfg.test_interval, cfg.total_fedbl_num, cfg.fedbl)
    api = Api(app)
    app.run(host=str(server_ip), port=int(sever_port), debug=False)

if __name__ == '__main__':
    args = parse_args()
    #生成分布式训练config
    start(args.config, args.ip, args.port)
