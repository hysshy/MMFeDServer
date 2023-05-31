import os.path
from mmcv import Config
from client import app
from flask import request, send_file
import pkg_resources
from utils.Log import logger
from utils.common import is_file_transfer_complete

def get_package_version(package_name):
    try:
        package = pkg_resources.get_distribution(package_name)
        return package.version
    except pkg_resources.DistributionNotFound:
        return None

@app.route('/download_file', methods=['POST'])
def download_file():
    # 检查是否有上传的文件
    if 'file' not in request.files:
        return 'No file uploaded', 400
    # 获取上传的文件对象
    file = request.files['file']
    # 检查文件名是否为空
    if file.filename == '':
        return 'Empty file name', 400
    # 处理文件，保存到本地文件系统
    savepath = request.form['savepath']
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    file.save(savepath+'/'+file.filename)
    logger.info('下载完毕文件:' + savepath+'/'+file.filename)
    # 返回响应
    return savepath+'/'+file.filename, 200

@app.route('/upload_file', methods=['POST'])
def upload_file():
    current_path = os.getcwd()
    filepath = current_path+'/'+request.form['filepath']
    if not os.path.exists(filepath):
        # 获取训练时间
        # init log path
        log_path = ''
        paths = filepath.split('/')
        for i in range(len(paths) - 1):
            log_path = log_path + paths[i] + '/'
        logfile = log_path + 'train.log'
        epoch_time = get_epoch_time(logfile)
        logger.info(':模型训练中,预计训练时间:'+str(epoch_time))
        return str(epoch_time), 400
    else:
        assert is_file_transfer_complete(filepath, 10)
        logger.info('上传模型:'+filepath)
        return send_file(filepath, as_attachment=True), 200


@app.route('/download_epoch', methods=['GET'])
def download_epoch():
    # 检查是否有上传的文件
    if 'file' not in request.files:
        return 'No file uploaded', 400
    # 获取cfg路径
    cfg_file = request.data.decode()
    cfg_files = cfg_file.split('/')
    job_dir = ''
    for i in range(len(cfg_files) - 2):
        job_dir = job_dir + cfg_files[i] + '/'
    # 获取上传的文件对象
    file = request.files['file']
    # 检查文件名是否为空
    if file.filename == '':
        return 'Empty file name', 400
    # 处理文件，保存到本地文件系统
    file.save(job_dir+'/'+file.filename)
    # 返回响应
    return job_dir+'/'+file.filename, 200

