
import os
import numpy as np
from parameters import *
import random
import glob
import cv2
import numpy as np

def create_file_list():
    filename = glob.glob('/home/zl/Data/Dual_Inception/3class_flow_data/*/*')

    sublist = []
    for i in range(len(filename)):
        if filename[i].split('/')[-1].split('_')[2] == 'smic':
            sublist.append(filename[i].split('/')[-1].split('_')[2] + '_'
                           + filename[i].split('/')[-1].split('_')[4])
        else:
            sublist.append(filename[i].split('/')[-1].split('_')[2] + '_' + filename[i].split('/')[-1].split('_')[3])

    sublist_quchong = list(set(sublist))
    sublist_quchong.sort()
    # train_sub = []
    # val_sub = []
    # test_sub = []
    # split = int(len(sublist_quchong)*0.2)
    # for i in range(len(sublist_quchong)):
    #     tmp_list = []
    #     tmp_list.append(sublist_quchong[i])
    #     test_sub.append(tmp_list)
    #     train_val = list(set(sublist_quchong)-set(tmp_list))
    #     tmp = random.sample(train_val,split)
    #     tmp_train = list(set(train_val)-set(tmp))
    #     val_sub.append(tmp)
    #     train_sub.append(tmp_train)
    train_sub = []
    val_sub = []
    test_sub = []
    split = int(len(sublist_quchong) * 0.2)
    for i in range(len(sublist_quchong)):
        tmp_list = []
        tmp_list.append(sublist_quchong[i])
        test_sub.append(tmp_list)
        train_val = list(set(sublist_quchong) - set(tmp_list))
        tmp = []
        for j in range(len(train_val)):
            if tmp_list[0].split('_')[0] == train_val[j].split('_')[0]:
                tmp.append(train_val[j])
        tmp_train = list(set(train_val) - set(tmp))
        val_sub.append(tmp)
        train_sub.append(tmp_train)
    return filename, train_sub, val_sub, test_sub

def read_file_directly(filename, test_subject, train_sub, val_sub, test_sub):
    fold = test_subject
    train_sub_list = []
    val_sub_list = []
    test_sub_list = []
    test_sample_list = []
    for i in range(len(filename)):
        # temp for smic_hs
        #temp1 for casme2 and samm
        tmp = filename[i].split('/')[-1].split('_')[2]+'_'+filename[i].split('/')[-1].split('_')[4]
        tmp1 = filename[i].split('/')[-1].split('_')[2]+'_'+filename[i].split('/')[-1].split('_')[3]
        for j in range(len(train_sub[fold])):
            if tmp==train_sub[fold][j]:
                train_sub_list.append(filename[i])
            if tmp1==train_sub[fold][j]:
                train_sub_list.append(filename[i])
        for k in range(len(val_sub[fold])):
            if tmp==val_sub[fold][k]:
                val_sub_list.append(filename[i])
            if tmp1==val_sub[fold][k]:
                val_sub_list.append(filename[i])
        for m in range(len(test_sub[fold])):
            if tmp==test_sub[fold][m]:
                test_sub_list.append(filename[i])
            if tmp1==test_sub[fold][m]:
                test_sub_list.append(filename[i])


    x_train_image_list = []
    y_train_image_list = []
    train_label_list = []
    for i in range(len(train_sub_list)):
        # print('train_img:',train_sub_list[i])
        img = cv2.imread(train_sub_list[i])
        img = cv2.resize(img,(28,28))
        img = np.array(img)
        img = img/255.0
        # print(train_sub_list[i].split('/'))
        tmp = int(train_sub_list[i].split('/')[-2])
        # print(train_sub_list[i].split('/')[7].split('_')[1])
        if train_sub_list[i].split('/')[-1].split('_')[1]=='x':
            x_train_image_list.append(img)
            train_label_list.append(tmp)
        else:
            y_train_image_list.append(img)
    x_val_image_list = []
    y_val_image_list = []
    val_label_list = []
    for i in range(len(val_sub_list)):
        # print('val_img:', val_sub_list[i])
        img = cv2.imread(val_sub_list[i])
        img = cv2.resize(img,(28,28))
        img = np.array(img)
        img = img/255.0
        tmp = int(val_sub_list[i].split('/')[-2])
        if val_sub_list[i].split('/')[-1].split('_')[1]=='x':
            x_val_image_list.append(img)
            val_label_list.append(tmp)
        else:
            y_val_image_list.append(img)
    x_test_image_list = []
    y_test_image_list = []
    test_label_list = []
    for i in range(len(test_sub_list)):
        # print('test_img:', test_sub_list[i])
        img = cv2.imread(test_sub_list[i])
        img = cv2.resize(img,(28,28))
        img = np.array(img)
        img = img/255.0
        tmp = int(test_sub_list[i].split('/')[-2])
        if test_sub_list[i].split('/')[-1].split('_')[1]=='x':

            test_sample_list.append(test_sub_list[i].split('/')[-1])

            x_test_image_list.append(img)
            test_label_list.append(tmp)
        else:
            y_test_image_list.append(img)
    train_x_array = np.array(x_train_image_list)
    train_y_array = np.array(y_train_image_list)
    train_label = np.array(train_label_list)
    val_x_array = np.array(x_val_image_list)
    val_y_array = np.array(y_val_image_list)
    val_label = np.array(val_label_list)
    test_x_array = np.array(x_test_image_list)
    test_y_array = np.array(y_test_image_list)
    test_label = np.array(test_label_list)
    return train_x_array, train_y_array, train_label, val_x_array, val_y_array, val_label, test_x_array, test_y_array, test_label, test_sample_list




# test subject: subject_test, set from 0 to 67
# other subjects: train
# get_train_test_data for parse train and test file array
# if 68 subjects then train 68 times





#
# if __name__ == '__main__':
#     # new_train()
#     image_process(path)

