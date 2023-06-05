import numpy as np
import math
def fed_BL(fedlw_num, cfg_list):
    w_loss_list = []
    g_loss_list = []
    for i in range(len(cfg_list)):
        print(cfg_list[i]['loss'])
        w_loss_list.append(cfg_list[i]['loss'][-1])
        g_loss_list.append(cfg_list[i]['loss'][-1]/cfg_list[i]['loss'][-2])
        # loss_i = cfg_list[i]['loss']
        # w_loss += loss_i
    # w_loss_mean = w_loss/len(cfg_list)
    # w_loss = np.mean(w_loss_list)
    # w_loss = np.max(w_loss_list)
    g_loss_mean = np.mean(g_loss_list)
    w_loss_mean = np.mean(w_loss_list)
    K = 10
    for i in range(len(cfg_list)):
        # if cfg_list[i]['client_cfg'].total_fedlw_num == fedlw_num:
        #     cfg_list[i]['fedlw'] = 1
        # else:
        # cfg_list[i]['fedlw'] = w_loss/cfg_list[i]['loss']
        # if g_loss_list[i] < g_loss_mean:
        #     w_i = g_loss_mean/g_loss_list[i]
        # else:
        #     w_i = 0
        g = K * math.exp()
        w = math.exp(max((w_loss_mean - w_loss_list[i]), 0)/0.15)
        # cfg_list[i]['fedlw'].append(((w_loss / cfg_list[i]['loss']))**0.5)
        cfg_list[i]['bl_w'].append(w)
        print(cfg_list[i]['bl_w'])

if __name__ == '__main__':
    print(math.exp(-2.12))
    print(math.exp(1.5))