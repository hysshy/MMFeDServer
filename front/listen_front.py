import os
# from flask_cors import CORS
from flask import request, send_file
from utils.Log import logger
from front import app
import requests
from flask_restful import Api
import copy
from mmcv import Config, ConfigDict
import json
import fedserver
import threading
netPath = '/home/chase/PycharmProjects/MMFeDServer/front/nets/'
taskPath = '/home/chase/PycharmProjects/MMFeDServer/front/tasks/'
basenet = '/home/chase/PycharmProjects/MMFeDServer/front/faster_rcnn_r50_fpn_2x_WiderFace_FedDGA.py'
jsonPath = '/home/chase/shy/dataset/publicDatas/lunwenjson/'

@app.route('/fedserver/is_available', methods=['POST'])
def is_available():
    data = request.get_json()
    ip = data['ip']
    port = data['port']
    logger.info('前段消息:检查客户端:'+ str(ip) + '-' + str(port))
    reponse = requests.post('http://{}:{}/check_online'.format(ip, port))
    if reponse.status_code != 200:
        logger.info('客户端不在线')
        return {'code':1, 'gpu':[], 'msg':[]}
    logger.info(reponse.json())
    return reponse.json()

@app.route('/fedserver/add_task', methods=['POST'])
def add_task():
    data = request.get_json()
    task_id = data['task_id']
    with open(taskPath+str(task_id)+'.json', "w") as f:
        json.dump(data, f)
    task_list = data['task_list']
    cfg = Config.fromfile(basenet)
    rpn_head = []
    roi_head = []
    data = []
    data_item_src = cfg.data[0]
    for i in range(len(task_list)):
        dataset = task_list[i]['dataset']
        gpu = task_list[i]['gpu']
        client_ip = task_list[i]['ip']
        client_port = task_list[i]['port']
        tasktype = dataset.split('_')[0]
        for i in range(len(cfg.model.rpn_head)):
            if cfg.model.rpn_head[i].rpn_head_type == tasktype and cfg.model.rpn_head[i] not in rpn_head:
                rpn_head.append(cfg.model.rpn_head[i])
        for i in range(len(cfg.model.roi_head)):
            if cfg.model.roi_head[i].roi_head_type == tasktype and cfg.model.roi_head[i] not in roi_head:
                roi_head.append(cfg.model.roi_head[i])
        data_item = copy.deepcopy(data_item_src)
        data_item.client_ip = client_ip
        data_item.client_port = client_port
        data_item.gpu_ids = [gpu]
        data_item.tasktype = tasktype
        data_item.train.ann_file = jsonPath+dataset
        data_item.test.ann_file = jsonPath+tasktype+'_test_res.json'
        data_item.val.ann_file = jsonPath+tasktype+'_test_res.json'
        data.append(data_item)
    cfg.model.rpn_head = rpn_head
    cfg.model.roi_head = roi_head
    cfg.data = data
    cfg.job_id = task_id
    cfg.dump(netPath+str(task_id)+'.py')
    return send_file(netPath+str(task_id)+'.py', as_attachment=True)

@app.route('/fedserver/get_task_list', methods=['POST'])
def get_task_list():
    taskdir = os.listdir(taskPath)
    respond = {'code':0}
    task_list = []
    for taskfile in taskdir:
        print(taskfile)
        with open(taskPath+taskfile, 'rb') as f:
            data = f.read()
            data = json.loads(data.decode())
            task_list.append(data)
    respond.setdefault('task_list', task_list)
    return respond

@app.route('/fedserver/get_loss', methods=['POST'])
def get_loss():
    data = request.get_json()
    task_id = data['task_id']
    losses = {"task_id":task_id}
    lossdict = {}
    with open(taskPath+str(task_id)+'.json', 'rb') as f:
        data = f.read()
        data = json.loads(data.decode())
        task_list = data['task_list']
        for i in range(len(task_list)):
            ip = task_list[i]['ip']
            port = task_list[i]['port']
            name = task_list[i]['name']
            logfile = 'job/'+str(task_id)+'/'+str(i)+'_'+str(ip)+'/'+'train.log'
            reponse = requests.get('http://{}:{}/get_client_loss'.format(ip, port), json={'logfile':logfile})
            loss = reponse.json()
            lossdict.setdefault(name, loss)
        losses.setdefault('loss', lossdict)
    return losses

@app.route('/fedserver/get_schedule', methods=['POST'])
def get_schedule():
    data = request.get_json()
    task_id = data['task_id']
    schedules = {"task_id":task_id}
    scheduledict = {}
    with open(taskPath+str(task_id)+'.json', 'rb') as f:
        data = f.read()
        data = json.loads(data.decode())
        task_list = data['task_list']
        for i in range(len(task_list)):
            ip = task_list[i]['ip']
            port = task_list[i]['port']
            name = task_list[i]['name']
            logfile = 'job/'+str(task_id)+'/'+str(i)+'_'+str(ip)+'/'+'train.log'
            reponse = requests.get('http://{}:{}/get_client_schedule'.format(ip, port), json={'logfile':logfile})
            schedule = reponse.text
            scheduledict.setdefault(name, schedule)
        schedules.setdefault('schedule', scheduledict)
    return schedules

@app.route('/fedserver/get_hardware', methods=['POST'])
def get_hardware():
    data = request.get_json()
    task_id = data['task_id']
    hardwares = {"task_id":task_id}
    hardwaredict = {}
    with open(taskPath+str(task_id)+'.json', 'rb') as f:
        data = f.read()
        data = json.loads(data.decode())
        task_list = data['task_list']
        for i in range(len(task_list)):
            ip = task_list[i]['ip']
            port = task_list[i]['port']
            name = task_list[i]['name']
            gpu = int(task_list[i]['gpu'])
            reponse = requests.get('http://{}:{}/get_client_hardware'.format(ip, port), json={'gpu':gpu})
            hardwaredict.setdefault(name, reponse.json())
        hardwares.setdefault('hardware', hardwaredict)
    return hardwares

@app.route('/fedserver/start_task', methods=['POST'])
def start_task():
    data = request.get_json()
    task_id = data['task_id']
    config = netPath + str(task_id)+'.py'
    t = threading.Thread(target=fedserver.start, args=(config, '0.0.0.0', 6000))
    t.start()
    reponse = {'code': 0, 'task_id':task_id, 'msg': '开始训练任务成功'}
    return reponse

@app.route('/fedserver/test', methods=['GET'])
def ddd():
    print('wwwwwwwwwwwwwww')
    print('dfdfdfdf')
    return 'jjj'

if __name__ == '__main__':
    # CORS(app,supports_credentials=True)
    api = Api(app)
    app.run(host='0.0.0.0', port=5001, debug=False)
    # cfg = Config.fromfile('/home/chase/PycharmProjects/MMFeDServer/front/faster_rcnn_r50_fpn_2x_WiderFace_FedDGA.py')
    # cfg.model.rpn_head.pop(0)
    # print(cfg.model.rpn_head)
    # print(1)