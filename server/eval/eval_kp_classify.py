import os
import time

from mmdet.datasets.api_wrappers import COCO
from mmdet.apis import init_detector, inference_detector
from mmcv import Config
import cv2
import numpy as np
from utils.Log import logger

colorList = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255)]
detect_categoriesName = ['person', 'lianglunche', 'sanlunche', 'car', 'truck', 'dog', 'cat']
faceGender_CategoriesName = ['male', 'female']

def kp_succesRate(pre_keypoints, gt_keypoints, gt_bboxes):
    success = np.zeros(len(pre_keypoints)*5)
    for i in range(len(pre_keypoints)):
        threshold = np.sqrt(gt_bboxes[i][2]**2 + gt_bboxes[i][2]**2) * 0.1
        for j in range(5):
            dist = np.sqrt((pre_keypoints[i][j][0]-gt_keypoints[i][j][0])**2 + (pre_keypoints[i][j][1]-gt_keypoints[i][j][1])**2)
            success[i*5+j] = int(dist <= threshold)
    return np.mean(success)

def label_accuracy(pre_labels, gt_labels):
    success = np.zeros(len(pre_labels))
    for i in range(len(pre_labels)):
        success[i] = int(pre_labels[i] == gt_labels[i]-1)
    return np.mean(success)

def infer_with_prebbox(config_file, checkpoint_file, adaptive_w_dict, gpu_id, type = 'faceKp', drawPath = None):

    cfg = Config.fromfile(config_file)
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:{}'.format(str(gpu_id)))
    dataset = COCO(cfg.data.test.ann_file)
    img_ids = dataset.get_img_ids()
    gt_bboxes, gt_labels, gt_keypoints, pre_bboxes, pre_labels, pre_keypoints = [],[],[],[],[],[]
    for id in img_ids:
        start = time.time()
        imgs = dataset.load_imgs(id)
        an_ids = dataset.get_ann_ids(img_ids=id)
        anlist = dataset.load_anns(an_ids)
        pre_bbox = []
        for an in anlist:
            bbox = an['bbox']
            pre_bbox.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], 0])
            gt_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            gt_keypoints.append(an['keypoints'])
            gt_labels.append(an['category_id'])
        results = inference_detector(model, imgs[0]['file_name'], pre_bbox, adaptive_w_dict)
        print(time.time() -start)
        img = cv2.imread(imgs[0]['file_name'])
        if type == 'faceKp':
            facekps = results
            for i in range(len(facekps)):
                pre_keypoints.append(facekps[i])
                faceKp = facekps[i].astype(int)
                for k in range(5):
                    point = (faceKp[k][0], faceKp[k][1])
                    cv2.circle(img, point, 3, (255, 0, 0), 0)
            if drawPath is not None:
                cv2.imwrite(drawPath+'/'+imgs[0]['file_name'].split('/')[-1], img)
        elif type == 'faceGender':
            faceGender_labels = results
            for i in range(len(faceGender_labels)):
                pre_labels.append(faceGender_labels[i])
                if drawPath is not None:
                    cv2.imwrite(drawPath+'/'+imgs[0]['file_name'].split('/')[-1].replace('.jpg', '_'+str(i)+'_'+faceGender_CategoriesName[faceGender_labels[i]]+'.jpg'), img[pre_bbox[i][1]:pre_bbox[i][3], pre_bbox[i][0]:pre_bbox[i][2]])
    if type == 'faceKp':
        kp_accuracy = kp_succesRate(pre_keypoints, gt_keypoints, gt_bboxes)
        logger.info('准确率测试 '+type+':'+str(kp_accuracy))
    elif type == 'faceGender':
        gender_accuracy = label_accuracy(pre_labels, gt_labels)
        logger.info('准确率测试 '+type+':'+str(gender_accuracy))
    return gt_bboxes, gt_labels, gt_keypoints, pre_bboxes, pre_labels, pre_keypoints


def infer(config_file, checkpoint_file, test_anfile, gpu_id, drawPath = None):
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:{}'.format(str(gpu_id)))
    dataset = COCO(test_anfile)
    img_ids = dataset.get_img_ids()
    for id in img_ids:
        imgs = dataset.load_imgs(id)
        result, labels, facekps  = inference_detector(model, imgs[0]['file_name'])
        img = cv2.imread(imgs[0]['file_name'])
        for i in range(len(result)):
            bbox = result[i].astype(int)
            if len(bbox) > 0:
                label = labels[i]
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i % len(colorList)], 1)
                cv2.putText(img, detect_categoriesName[label], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            colorList[i % len(colorList)], 2)
            faceKp = facekps[i].astype(int)
            for k in range(5):
                point = (faceKp[k][0], faceKp[k][1])
                cv2.circle(img, point, 3, (255, 0, 0), 0)
        if drawPath is not None:
            cv2.imwrite(drawPath+'/'+imgs[0]['file_name'].split('/')[-1], img)


if __name__ == '__main__':

    # config_file = '/home/chase/shy/mutiltask_mmdetection/work_dirs/faster_rcnn_r50_fpn_2x_coco_spjc2/faster_rcnn_r50_fpn_2x_spjc.py'
    # checkpoint_file = '/home/chase/shy/mutiltask_mmdetection/work_dirs/faster_rcnn_r50_fpn_2x_coco_spjc2/epoch_12.pth'
    # test_anfile = '/home/chase/shy/dataset/spjgh/lunwenjson2/faceGender_test_res.json'
    # drawPath = '/home/chase/shy/dataset/spjgh/draw'
    # gpu_id = 0
    #
    # gt_bboxes, gt_labels, gt_keypoints, pre_bboxes, pre_labels, pre_keypoints = infer_with_prebbox(config_file, checkpoint_file, 0, 'faceGender', drawPath=None)
    # kp_accuracy = kp_succesRate(pre_keypoints, gt_keypoints, gt_bboxes)
    # # print(kp_accuracy)
    # gender_accuracy = label_accuracy(pre_labels, gt_labels)
    # print(gender_accuracy)

    # dataset = COCO('/home/chase/shy/dataset/spjgh/test_res.json')
    # img_ids = dataset.get_img_ids()
    # imgs = dataset.load_imgs(img_ids)
    # an_ids = dataset.get_ann_ids(img_ids=1)
    # an = dataset.load_anns(an_ids)
    # print(an)

    config_file = '/home/chase/PycharmProjects/MMFedClient/job/singletask/faster_rcnn_r50_fpn_2x_WiderFace-faceKp.py'
    checkpoint_file = '/home/chase/PycharmProjects/MMFedClient/job/singletask/WiderFace-kp.pth'
    infer_with_prebbox(config_file, checkpoint_file, None, 0, type = 'faceGender', drawPath = None)