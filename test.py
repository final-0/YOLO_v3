from model import *
from data import *
from data_arrange import *
from vision1 import *
from anchor import *
import torchvision.transforms as T
import torch
import os
import glob
import pandas as pd
from sklearn.cluster import KMeans
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from torchsummary import summary
from PIL import ImageDraw, ImageFont, Image

coco128 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush'] 

coco =  ['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 
        'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 
        'cat', 'cell phone', 'chair', 'clock', 'cow', 'cup', 'diningtable', 'dog', 'donut', 'elephant', 
        'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier', 'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 
        'knife', 'laptop', 'microwave', 'motorbike', 'mouse', 'orange', 'oven', 'parking meter', 'person', 'pizza', 
        'pottedplant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 
        'sofa', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 
        'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tvmonitor', 'umbrella', 'vase', 'wine glass', 'zebra']

os.makedirs("test_images", exist_ok=True)

#=== preparation ===#
class_n = 80
model = YOLOv3().to('cuda')
optimizer = torch.optim.Adam(model.parameters())
anchor = get_anchor(Str="../data/coco/valid/labels")
img_list = sorted(glob.glob(os.path.join("../data/coco/valid/images","*")))
transform = T.Compose([T.ToTensor()])
img_size = 416

model_path = 'model_va_500.pth'
model.load_state_dict(torch.load(model_path))
model = model.cuda()
summary(model,(3,416,416))

#=== test ===#
num = 0
for path in img_list:
    num += 1
    if num % 5 == 0:
        img = cv2.imread(path)
        img = cv2.resize(img , (img_size , img_size))
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            preds  = list(model(img))
        img = Image.open(path)   
        img = img.resize((img_size , img_size))
        draw = ImageDraw.Draw(img, mode="RGBA")
        for color,pred in zip(["red","orange","blue"],preds):
            place = vision(pred,anchor,img_size,conf=0.9)
            for l in range(len(place)):
                P = place[l]
                P = np.array(P)
                P = np.clip(P,0,img_size)
                P = P.tolist()
                
                x1,y1,x2,y2,max_n,max_c = P[0],P[1],P[2],P[3],P[4],P[5]
                max_c = coco[int(max_c)]
                draw.rectangle((x1,y1,x2,y2), outline=color, width=2)
                if x2-x1 >= len(max_c)*6+2:
                    draw.rectangle((x1,y1,x1+len(max_c)*6+1,y1+11) ,fill=color ,outline=color)
                    draw.text((x1+1,y1+1), max_c, fill="white")

        img.save("test_images/%d.png" % num)