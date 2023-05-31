import copy

import torch

if __name__ == '__main__':
    backbone_cpt = '/home/chase/PycharmProjects/MMFedClient/job/2/merge_epoch_30.pth'
    detect_cpt = '/home/chase/PycharmProjects/MMFedClient/job/2/0_10.10.5.136/epoch_30.pth'
    merge_cpt = '/home/chase/PycharmProjects/MMFedClient/job/2/0_10.10.5.136/epoch_merge.pth'
    backbone_model = torch.load(backbone_cpt, map_location='cpu')
    detect_model = torch.load(detect_cpt, map_location='cpu')

    merge_model = copy.deepcopy(detect_model)
    for key in merge_model['state_dict'].keys():
        if key.split('.')[0] in ['backbone', 'neck', 'rpn_head']:
            merge_model['state_dict'][key] = backbone_model[key]
        elif key.split('.')[0] in ['detect_roi_head']:
            merge_model['state_dict'][key] = detect_model['state_dict'][key]
    torch.save(merge_model, merge_cpt)