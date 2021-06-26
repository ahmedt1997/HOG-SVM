# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 03:14:25 2021

@author: Med Chlif
"""

import numpy as np 
import matplotlib.pyplot as plt 
import glob
import cv2
import os
from skimage.feature import hog

import seaborn as sns


print(os.listdir("dataset"))

train_images = []
train_labels = []

canvas_hog = []
cushion1_hog = []


for directory_path in glob.glob("dataset/training_set/canvas1"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        canvas_hog.append(fd)
        X_canvas=np.vstack(canvas_hog).astype(np.float64)
        y_canvas=np.ones(len(X_canvas))
        


for directory_path in glob.glob("dataset/training_set/cushion1"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        cushion1_hog.append(fd)
        X_cushion1=np.vstack(cushion1_hog).astype(np.float64)
        y_cushion1=np.full(len(X_cushion1),0)


linsseeds1_hog = []

for directory_path in glob.glob("dataset/training_set/linsseeds1"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        linsseeds1_hog.append(fd)
        X_linsseeds1=np.vstack(linsseeds1_hog).astype(np.float64)
        y_linsseeds1=np.full(len(X_linsseeds1),2)



sand1_hog = []

for directory_path in glob.glob("dataset/training_set/sand1"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        sand1_hog.append(fd)
        X_sand1=np.vstack(sand1_hog).astype(np.float64)
        y_sand1=np.full(len(X_sand1),3)


seat2_hog = []

for directory_path in glob.glob("dataset/training_set/seat2"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        seat2_hog.append(fd)
        X_seat2=np.vstack(seat2_hog).astype(np.float64)
        y_seat2=np.full(len(X_seat2),4)


stone1_hog = []


for directory_path in glob.glob("dataset/training_set/stone1"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        stone1_hog.append(fd)
        X_stone1=np.vstack(stone1_hog).astype(np.float64)
        y_stone1=np.full(len(X_stone1),5)


X_training = np.vstack((X_canvas,X_cushion1,X_linsseeds1,X_sand1,X_seat2,X_stone1))
y_training = np.hstack((y_canvas,y_cushion1,y_linsseeds1,y_sand1,y_seat2,y_stone1))


########################### preprocessing for test data ####################3


canvas_hog_test = []
cushion1_hog_test = []
linsseeds1_hog_test = []
sand1_hog_test = []
seat2_hog_test = []
stone1_hog_test = []


for directory_path in glob.glob("dataset/test_set/canvas1"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        
        canvas_hog_test.append(fd)
        X_canvas_test=np.vstack(canvas_hog_test).astype(np.float64)
        y_canvas_test=np.ones(len(X_canvas_test))


for directory_path in glob.glob("dataset/test_set/cushion1"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        cushion1_hog_test.append(fd)
        X_cushion1_test=np.vstack(cushion1_hog_test).astype(np.float64)
        y_cushion1_test=np.full(len(X_cushion1_test),0)



for directory_path in glob.glob("dataset/test_set/linsseeds1"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        linsseeds1_hog_test.append(fd)
        X_linsseeds1_test=np.vstack(linsseeds1_hog_test).astype(np.float64)
        y_linsseeds1_test=np.full(len(X_linsseeds1_test),2)



for directory_path in glob.glob("dataset/test_set/sand1"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        sand1_hog_test.append(fd)
        X_sand1_test=np.vstack(sand1_hog_test).astype(np.float64)
        y_sand1_test=np.full(len(X_sand1_test),3)




for directory_path in glob.glob("dataset/test_set/seat2"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        seat2_hog_test.append(fd)
        X_seat2_test=np.vstack(seat2_hog_test).astype(np.float64)
        y_seat2_test=np.full(len(X_seat2_test),4)



for directory_path in glob.glob("dataset/test_set/stone1"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (64, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
        stone1_hog_test.append(fd)
        X_stone1_test=np.vstack(stone1_hog_test).astype(np.float64)
        y_stone1_test=np.full(len(X_stone1_test),5)




X_test = np.vstack((X_canvas_test,X_cushion1_test,X_linsseeds1_test,X_sand1_test,X_seat2_test,X_stone1_test))
y_test = np.hstack((y_canvas_test,y_cushion1_test,y_linsseeds1_test,y_sand1_test,y_seat2_test,y_stone1_test))




from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix

svc_model = LinearSVC()

svc_model.fit(X_training,y_training)


y_predict = svc_model.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix

cm = confusion_matrix(y_test,y_predict)
        
sns.heatmap(cm, annot=True,fmt='d')


print(classification_report( y_test, y_predict))
   
# image_test=cv2.imread('canvas1-a-p001.png', cv2.IMREAD_COLOR)   

# image_test = cv2.resize(image_test, (64, 128))
# image_test = cv2.cvtColor(image_test, cv2.COLOR_RGB2BGR)

# canvas_hog1 = []

# #creating hog features 
# fd1, hog_image1 = hog(image_test, orientations=9, pixels_per_cell=(8, 8), 
#                     cells_per_block=(2, 2), visualize=True, multichannel=True)   

# canvas_hog1.append(fd1)

# X1=np.vstack(canvas_hog1).astype(np.float64)

# y1=np.ones(len(X1))
 
   
        



        
        # train_images.append(img)
        # train_labels.append(label)



        
        
        # canvas_hog.append(fd)
        # X=np.vstack(canvas_hog).astype(np.float64)
        # y=np.ones(len(X))
        




#         train_images.append(img)
#         train_labels.append(label)
        
# train_images = np.array(train_images)
# train_labels = np.array(train_labels)


# # test
# test_images = []
# test_labels = [] 
# for directory_path in glob.glob("dataset/test_set/*"):
#     fruit_label = directory_path.split("\\")[-1]
#     for img_path in glob.glob(os.path.join(directory_path, "*.png")):
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, (size, size))
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         test_images.append(img)
#         test_labels.append(fruit_label)
        
# test_images = np.array(test_images)
# test_labels = np.array(test_labels)


# #Encode labels from text to integers.
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()

# le.fit(test_labels)
# test_labels_encoded = le.transform(test_labels)

# le.fit(train_labels)
# train_labels_encoded = le.transform(train_labels)



# plt.imshow(train_images[6])
