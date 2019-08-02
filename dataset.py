import os
import cv2 as cv
import numpy as np

def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 文件夹下的所有文件名
    file_name = []
    # 载入数据路径并写入标签值
    # list_dir = os.listdir(file_dir)
    for file in os.listdir(file_dir):
        img_dir = os.path.join(file_dir,file)
        file_name.append(img_dir)
    nub = str(len(file_name))
    print('data number is '+nub)
    return file_name

def load_imag(path):
    img = cv.imread(path)
    img = img[:, :, ::-1]
    img = cv.resize(img, (300, 300))
    return img

def get_data(path,min=None):
    imgs_dir = get_files(path)
    if min != None:
        imgs_dir = imgs_dir[:min]
    input = []
    labels = []
    for i in range(len(imgs_dir)):
        file = imgs_dir[i]
        img = load_imag(file)
        if "dog" in file:
            label = [1]
        else:
            label = [0]
        # img = img/255.
        input += [img]
        labels += [label]
        if i%100==0:
            print('data load...: '+str(i))
    input_seq = np.asarray(input)
    target_seq = np.asarray(labels)
    # cv.namedWindow('a',0)
    # cv.imshow('a',input_seq[0][:])
    # cv.waitKey(0)
    return input_seq,target_seq

def random_batch(x_data,y_data,batch_size):
    rnd_indices = np.random.randint(0, len(x_data), batch_size)
    x_batch = x_data[rnd_indices][:]
    y_batch = y_data[rnd_indices][:]
    return x_batch, y_batch