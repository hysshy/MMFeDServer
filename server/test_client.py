from server.eval import eval_kp_classify, eval_bbox

def eval(client_cfg_list):
    finishtype = []
    for i in range(len(client_cfg_list)):
        if client_cfg_list[i]['tasktype'] in finishtype:
            continue
        else:
            finishtype.append(client_cfg_list[i]['tasktype'])
        if client_cfg_list[i]['tasktype'] in ['faceKp', 'faceGender']:
            eval_kp_classify.infer_with_prebbox(client_cfg_list[i]['client_cfg_file'],
                                                client_cfg_list[i]['merge_file'],
                                                {client_cfg_list[i]['tasktype']:None},
                                                0,
                                                client_cfg_list[i]['tasktype'])
        elif client_cfg_list[i]['tasktype'] in ['detect', 'carplateDetect', 'faceDetect']:
            print(client_cfg_list[i]['client_cfg_file'])
            print(client_cfg_list[i]['merge_file'])
            print(client_cfg_list[i]['tasktype'])
            eval_bbox.eval(client_cfg_list[i]['client_cfg_file'],
                           client_cfg_list[i]['merge_file'],
                           {client_cfg_list[i]['tasktype']: None},
                           tasktype=client_cfg_list[i]['tasktype'])


if __name__ == '__main__':
    eval_bbox.eval('/home/chase/PycharmProjects/MMFeDServer/job/98/0_10.10.7.201/faster_rcnn_r50_fpn_2x_spjc.py',
                   '/home/chase/PycharmProjects/MMFeDServer/job/98/merge_epoch_1.pth',
                   tasktype='detect')