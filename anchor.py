import os
import glob
import pandas as pd
from sklearn.cluster import KMeans

def get_anchor(Str):
    img_size = 416
    label_list = glob.glob(os.path.join(Str,"*"))
    label_list = sorted(label_list)

    box = {'width':[] , 'height':[]}
    box_list = []
    for path in label_list:
        with open(path , 'r',newline='\n') as f:
            for s_line in f:
                bbox = [float(x) for x in s_line.rstrip('\n').split(' ')]
                box['width'].append(bbox[3])
                box['height'].append(bbox[4])
                box_list.append(bbox[3:5])

    df = pd.DataFrame(box)
    km = KMeans(n_clusters=9,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
    df['cluster'] = km.fit_predict(box_list)

    anchor_dict = {"width":[],"height":[],"area":[]}
    for i in range(9):
        anchor_dict["width"].append(df[df["cluster"] == i].mean()["width"])
        anchor_dict["height"].append(df[df["cluster"] == i].mean()["height"])
        anchor_dict["area"].append(df[df["cluster"] == i].mean()["width"]*df[df["cluster"] == i].mean()["height"])
    
    anchor = pd.DataFrame(anchor_dict).sort_values('area', ascending=False)
    anchor["type"] = [int(img_size/32) ,int(img_size/32) ,int(img_size/32) ,  int(img_size/16) ,int(img_size/16) ,int(img_size/16) , int(img_size/8), int(img_size/8), int(img_size/8)]

    return anchor