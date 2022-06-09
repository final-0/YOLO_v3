#=== remove error labels and images ===#
def remove(files):
    label_remove = []
    img_remove = []
    for file in files:
        f = open(file, 'r')
        datalist = f.readlines()
        for data in datalist:
            each = data.split(" ")
            for i in range(len(each)):
                if i == 0:
                    continue
                else:
                    if float(each[i]) < 0.0000000000000000001:
                        if file not in label_remove:
                            label_remove.append(file)
                        if file.replace("txt","jpg").replace("labels","images") not in img_remove:
                            img_remove.append(file.replace(".txt",".jpg").replace("labels","images"))
        f.close()
    return label_remove, img_remove