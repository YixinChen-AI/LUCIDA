import torch.nn as nn
import torch
import torch.nn.functional as F

def cal_diceloss3d(pred_sm,y,is_y_onehot=False,logits=True):
    if logits:
        pred_sm = torch.sigmoid(pred_sm)
    if is_y_onehot:
        y_onehot = y
    else:
        y_onehot = torch.nn.functional.one_hot(y,num_classes = pred.shape[1]).permute(0,4,1,2,3)
    a = torch.sum(pred_sm*y_onehot,dim=[2,3,4])
    b = torch.sum(pred_sm,dim=[2,3,4]) + torch.sum(y_onehot,dim=[2,3,4])+1
    aa = torch.sum((1-pred_sm)*(1-y_onehot),dim=[2,3,4])
    bb = torch.sum((1-pred_sm),dim=[2,3,4]) + torch.sum((1-y_onehot),dim=[2,3,4])+1
    mask = torch.sum(y_onehot,dim=[2,3,4])
    mask[mask != 0] = 1
    dice = 2 * a / b
    lossdice = 1-2*a / b

    # rm backgroun
    lossdice = lossdice[:,:]
    mask = mask[:,:]
    return lossdice[mask == 1]