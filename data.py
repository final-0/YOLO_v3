from six import b
from torch.utils.data import Dataset,DataLoader
import cv2
from PIL import Image
import math
import torch
import numpy as np
from torchvision.ops.boxes import box_iou

class YOLOv3_Dataset(Dataset):
    def __init__(self,img_list , label_list,class_n,img_size,anchor_dict,transform):
        super().__init__()
        self.img_list = img_list
        self.label_list = label_list
        self.anchor_dict = anchor_dict
        self.class_n = class_n
        self.img_size = img_size
        self.transform = transform
        self.anchor_iou = torch.cat([torch.zeros(9,2) , torch.tensor(self.anchor_dict[["width","height"]].values)] ,dim = 1)

    def get_label(self , path):
        bbox_list = []
        with open(path , 'r',newline='\n') as f:  
            for s_line in f:
                bbox = [float(x) for x in s_line.rstrip('\n').split(' ')]
                bbox_list.append(bbox)
        return bbox_list
  
    def wh2twth(self, wh):
        twth = []
        for i in range(9):
            anchor = self.anchor_dict.iloc[i]
            aw = anchor["width"]
            ah = anchor["height"]
            twth.append([math.log(wh[0]/aw) , math.log(wh[1]/ah)])
        return twth

    def cxcy2txty(self,cxcy):
        map_size = [int(self.img_size/32) , int(self.img_size/16) , int(self.img_size/8)]
        txty = []
        for size in map_size:
            grid_x = int(cxcy[0]*size)
            grid_y = int(cxcy[1]*size)
            
            tx = math.log((cxcy[0]*size - grid_x + 1e-10) / (1 - cxcy[0]*size +grid_x+ 1e-10))
            ty = math.log((cxcy[1]*size - grid_y+ 1e-10) / (1 - cxcy[1]*size + grid_y+ 1e-10))
            txty.append([grid_x , tx , grid_y ,ty])
        return txty

    def label2tensor(self , bbox_list):
        map_size = [int(self.img_size/32) , int(self.img_size/16) , int(self.img_size/8)]
        tensor_list = []
    
        for size in map_size:
            for x in range(3):
                tensor_list.append(torch.zeros((4 + 1 + self.class_n,size,size)))
    
        for bbox in bbox_list:
            cls_n = int(bbox[0])
            txty_list = self.cxcy2txty(bbox[1:3])
            twth_list = self.wh2twth(bbox[3:])
            label_iou = torch.cat([torch.zeros((1,2))  , torch.tensor(bbox[3:]).unsqueeze(0)],dim=1)
            iou = box_iou(label_iou, self.anchor_iou)[0]
            obj_idx = torch.argmax(iou).item()
            for i , twth in enumerate(twth_list):
                tensor = tensor_list[i]
                txty = txty_list[int(i/3)]
                if i == obj_idx:
                    tensor[0,txty[2],txty[0]] = txty[1]
                    tensor[1,txty[2],txty[0]] = txty[3]
                    tensor[2,txty[2],txty[0]] = twth[0]
                    tensor[3,txty[2],txty[0]] = twth[1]
                    tensor[4,txty[2],txty[0]] = 1
                    tensor[5 + cls_n,txty[2],txty[0]] = 1
    

        scale3_label = torch.cat(tensor_list[0:3] , dim = 0)
        scale2_label = torch.cat(tensor_list[3:6] , dim = 0)
        scale1_label = torch.cat(tensor_list[6:] , dim = 0)

        return scale3_label , scale2_label , scale1_label  
     

    def __getitem__(self , idx):
        img_path = self.img_list[idx]
    
        label_path = self.label_list[idx]
    
        bbox_list = self.get_label(label_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img , (self.img_size , self.img_size))
        img = Image.fromarray(img)
        img = self.transform(img)
        scale3_label , scale2_label , scale1_label = self.label2tensor(bbox_list)
    
        return img , scale3_label , scale2_label , scale1_label
    
    def __len__(self):
        return len(self.img_list)