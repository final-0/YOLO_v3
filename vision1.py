import torch
import numpy as np
import matplotlib.pyplot as plt

class_n = 80
def vision(pred,anchor,img_size,conf = 0.5):
    size = pred.shape[2]
    anchor_size = anchor[anchor["type"] == size]
    place = []
    for i in range(3):
        a = anchor_size.iloc[i]
        grid = img_size/size
        pred = pred.detach()
        anc_pred = pred[0,i*(5+class_n):(i+1)*(5+class_n)].cpu()
        prob = torch.sigmoid(anc_pred[4,:,:]).cpu().numpy()
        index = np.where(prob > conf)
    
        for y,x in zip(index[0],index[1]):
            cx = x*grid + torch.sigmoid(anc_pred[0,y,x]).numpy()*grid
            cy = y*grid + torch.sigmoid(anc_pred[1,y,x]).numpy()*grid
            width = a["width"]*torch.exp(anc_pred[2,y,x]).numpy()*img_size
            height = a["height"]*torch.exp(anc_pred[3,y,x]).numpy()*img_size
            pr = anc_pred[5:,y,x].numpy()
            pr = pr.tolist()
            max_num = max(pr)
            max_class = pr.index(max_num)
            x1,y1,x2,y2 = cx - width/2 , cy - height/2 ,cx + width/2 , cy + height/2
            place.append([x1,y1,x2,y2,max_num,max_class])
    return place

#=====LOSS FUNCTION=====#
criterion_bce = torch.nn.BCEWithLogitsLoss()
criterion_mse = torch.nn.MSELoss()

def loss_F(pred , y_true,class_n = 80):
    loss_T = 0
    loss_F = 0
    loss_P = 0
    loss_C = 0
    for i in range(3):
        anc_pred = pred[:,i*(5+class_n) :(i+1)*(5+class_n) ]
        y_true_cut = y_true[:,i*(5+class_n) :(i+1)*(5+class_n) ]
        s = anc_pred.size()[2]
        loss_coord = torch.sum(torch.square(anc_pred[:,0:4] - y_true_cut[:,0:4])*y_true_cut[:,4])
        loss_t =  torch.sum(-1 * torch.log(torch.sigmoid(anc_pred[:,4])+ 1e-10)*(y_true_cut[:,4]))*s*s
        loss_f =  torch.sum(-1 * torch.log(1 - torch.sigmoid(anc_pred[:,4])+ 1e-10)*(1 - y_true_cut[:,4]))
        loss_class = torch.sum(torch.square(anc_pred[:,5:] - y_true_cut[:,5:])*y_true_cut[:,4])
        loss_T += loss_t
        loss_F += loss_f
        loss_P += loss_coord
        loss_C += loss_class

    return loss_P , loss_T , loss_F, loss_C