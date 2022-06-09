from model import *
from data import *
from data_arrange import *
from vision1 import *
from anchor import *
from history import *
import torchvision.transforms as T
from torchsummary import summary
import torch
import os
import glob
#"../data/coco/train/images"
#"../data/coco/valid/images"
#"../data/coco128/images/train2017"
#"../data/coco200/images"

#=== preparation ===#
class_n = 80
model = YOLOv3().to('cuda')
summary(model,(3,416,416))
optimizer1 = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer2 = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer3 = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer4 = torch.optim.Adam(model.parameters(), lr=0.00001)

anchor = get_anchor(Str="../data/coco/train/labels")
img_list = glob.glob(os.path.join("../data/coco/train/images","*"))
label_list = glob.glob(os.path.join("../data/coco/train/labels","*"))
label_remove, img_remove = remove(label_list)
for lis in label_remove:
    label_list.remove(lis)
for file in img_list:
    for remove in img_remove:
        if str(file) == str(remove):
            img_list.remove(file)
img_list = sorted(img_list)
label_list = sorted(label_list)

transform = T.Compose([T.ToTensor()])
img_size = 416
train_data = YOLOv3_Dataset(img_list,label_list,80,img_size , anchor,transform)
train_loader = DataLoader(train_data , batch_size = 1)
best_loss = 99999
Loss_history = []
print(len(train_loader))
#=== training ===#
for epoch in range(500):
    Train_loss = 0
    for n , (img , scale3_label , scale2_label ,scale1_label) in enumerate(train_loader):
        if epoch < 150:
            optimizer = optimizer2
        elif 150 <= epoch and epoch < 300:
            optimizer = optimizer3
        elif 300 <= epoch and epoch < 450:
            optimizer = optimizer4
        else:
            potimizer = optimizer4
        
        optimizer.zero_grad()
        img = img.cuda()
        scale1_label = scale1_label.cuda()
        scale2_label = scale2_label.cuda()
        scale3_label = scale3_label.cuda()
        labels = [scale3_label , scale2_label ,scale1_label]
        preds  = list(model(img))
            
        Loss = 0
        L1, L2, L3, L4 = 0,0,0,0
        for label , pred in zip(labels , preds):
            l1, l2, l3, l4 = loss_F(pred , label)
            Loss += l1
            L1 += l1
            Loss += l2
            L2 += l2
            Loss += l3
            L3 += l3
            Loss += l4
            L4 += l4
        Train_loss += Loss.item()
        Loss.backward()
        optimizer.step()
        print("[Epoch] %d, [batch] %d, [loss] %f :::: [T loss] %f, [F loss] %f, [place loss] %f, [class loss] %f" % (epoch, n, Train_loss/(n+1), L2, L3, L1, L4))
    
    Loss_history.append(Train_loss/(n+1))
    if best_loss > Train_loss/(n+1):
        model_path = 'model_tr_500.pth'
        torch.save(model.state_dict(), model_path)
        best_loss = Train_loss/(n+1)
plot_history(Loss_history)