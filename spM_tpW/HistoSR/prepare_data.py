import pickle
import tensorlayer as tl
import numpy as np
import os
import nibabel as nib
import cv2

training_data_path = "./dataset/MRI/MICCAI_2013_challenge/MICCAI-2013-SATA-Challenge-Data-Std-Reg/diencephalon/training-training/warped-images/"
testing_data_path = "./dataset/MRI/MICCAI_2013_challenge/MICCAI-2013-SATA-Challenge-Data-Std-Reg/diencephalon/training-training/warped-images/"
val_ratio = 0.3
seed = 100
preserving_ratio = 0.1 # filter out 2d images containing < 10% non-zeros
data_saving_path = './dataset/MRI/MICCAI_2013_challenge/preprocessed/'
tl.files.exists_or_mkdir(data_saving_path)

# dump training images
f_train_all = []
for line in open('DAGAN_Training_datasets.txt').readlines():
    for l in line.split():
        print(l)
        f_train_all.append(l)


train_all_num = len(f_train_all)
val_num = int(train_all_num * val_ratio)

f_train = []
f_val = []

val_idex = tl.utils.get_random_int(min_v=0,
                                   max_v=train_all_num - 1,
                                   number=val_num,
                                   seed=seed)
for i in range(train_all_num):
    if i in val_idex:
        f_val.append(f_train_all[i])
    else:
        f_train.append(f_train_all[i])

train_3d_num, val_3d_num = len(f_train), len(f_val)
print('number of training volumes: ', train_3d_num)

X_train = []
count = 0
train_image_path = data_saving_path + '/train/'
tl.files.exists_or_mkdir(train_image_path)
fw = open(data_saving_path+'/train.txt', 'w')
for fi, f in enumerate(f_train):
    print("processing [{}/{}] 3d image ({}) for training set ...".format(fi + 1, train_3d_num, f))
    img_path = os.path.join(training_data_path, f)
    img = nib.load(img_path).get_data()
    img_3d_max = np.max(img)
    img = img / img_3d_max * 255
    for i in range(img.shape[2]):
        img_2d = img[:, :, i]
        # filter out 2d images containing < 10% non-zeros
        if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:
            img_2d = np.transpose(img_2d, (1, 0))
            img_name = str(fi) + '_' + str(i) + '.bmp'
            cv2.imwrite(train_image_path + img_name, img_2d)
            fw.write(img_name+'\n')
            count += 1
print('number of training images: ', count)

# dump test images
f_test = []
for line in open('DAGAN_Testing_datasets.txt').readlines():
    for l in line.split():
        print(l)
        f_test.append(l)
test_3d_num = len(f_test)

X_test = []
test_image_path = data_saving_path + '/test/'
tl.files.exists_or_mkdir(test_image_path)
fw = open(data_saving_path + '/test.txt', 'w')
for fi, f in enumerate(f_test):
    print("processing [{}/{}] 3d image ({}) for test set ...".format(fi + 1, test_3d_num, f))
    img_path = os.path.join(testing_data_path, f)
    img = nib.load(img_path).get_data()
    img_3d_max = np.max(img)
    img = img / img_3d_max * 255
    for i in range(img.shape[2]):
        img_2d = img[:, :, i]
        # filter out 2d images containing < 10% non-zeros
        if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:
            img_2d = np.transpose(img_2d, (1, 0))
            X_test.append(img_2d)

X_test = np.asarray(X_test)
X_test = X_test[:, :, :, np.newaxis]
idex = tl.utils.get_random_int(min_v=0, max_v=len(X_test) - 1, number=50, seed=100)
X = X_test[idex]
for i in range(X.shape[0]):
    cv2.imwrite(test_image_path + str(i) + '.bmp', X[i, :, :, 0])
    fw.write(str(i) + '.bmp\n')
