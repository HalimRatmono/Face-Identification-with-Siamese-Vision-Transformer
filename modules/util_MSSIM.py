import os
import random

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

random.seed(15)
np.random.seed(15)
IMAGE_DIMS = 224

def preprocess(img, size=IMAGE_DIMS, interpolation =cv2.INTER_AREA):
    #extract image size
    h, w = img.shape[:2]
    #check color channels
    c = None if len(img.shape) < 3 else img.shape[2]
    #square images have an aspect ratio of 1:1
    if h == w: 
        return cv2.resize(img, (size, size), interpolation)
    elif h>w:#height is larger
        diff= h-w
        img=cv2.copyMakeBorder(img,0,0,int(diff/2.0),int(diff/2.0),cv2.BORDER_CONSTANT, value = 0)
        # img=cv2.copyMakeBorder(img,0,0,int(diff/2.0),int(diff/2.0),cv2.BORDER_REPLICATE)
    elif h<w:
        diff= w-h
        # img=cv2.copyMakeBorder(img,int(diff/2.0),int(diff/2.0),0,0,cv2.BORDER_REPLICATE)
        img=cv2.copyMakeBorder(img,int(diff/2.0),int(diff/2.0),0,0,cv2.BORDER_CONSTANT, value = 0)
    return cv2.resize(img, (size, size), interpolation)


def scale_back(img, size=IMAGE_DIMS):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  dif = size-h
  x_pos = int((dif)/2.0)
  y_pos = int((dif)/2.0)
  mask = np.zeros((size, size, c), dtype=img.dtype)
  mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
#   print(mask.shape)
  # cv2.imshow("test",mask)
  # cv2.waitKey(0)
  return mask

def mse(img1, img2):
    assert img1.shape == img2.shape
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])
	# return the MSE, the lower the error, the more "similar"
	# the two images are
    return err

def get_small_mse_img_in_cls_path(img1_path,class_path):
    """
    calculates all the mse difference between img1_path and other images in the class path
    keep the lowest 5 mse difference images and randomly return 1 of them
    """
    img1 = preprocess(img=cv2.imread(img1_path))
    lowest_5_mse_tuple = [] * 5
    for img2_path in os.listdir(class_path):
        img2_path = os.path.join(class_path,img2_path)
        if img2_path != img1_path:
            img2 = preprocess(img=cv2.imread(img2_path))
            mse_diff = mse(img1=img1,img2=img2)
            lowest_5_mse_tuple.append((img2_path,mse_diff))
            if len(lowest_5_mse_tuple) > 5:
                lowest_5_mse_tuple.sort(key=lambda x:x[1])
                del lowest_5_mse_tuple[-1]
                assert len(lowest_5_mse_tuple) == 5, "length of the tuple have to be 5"
    assert len(lowest_5_mse_tuple) <= 5, "length of the tuple have to less or equal to 5"
    return random.choice(lowest_5_mse_tuple)[0]

def get_high_ssim_img_in_cls_path(img1_path,class_path):
    """
    calculates all the mssim difference between img1_path and other images in the class path
    keep the highest 5 mse difference images and randomly return 1 of them
    """
    img1 = preprocess(img=cv2.imread(img1_path))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    highest_5_ssim_tuple = [] * 5
    for img2_path in os.listdir(class_path):
        img2_path = os.path.join(class_path,img2_path)
        if img2_path != img1_path:
            img2 = preprocess(img=cv2.imread(img2_path))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            ssmi = ssim(img1,img2)
            highest_5_ssim_tuple.append((img2_path,ssmi))
            if len(highest_5_ssim_tuple) > 5:
                highest_5_ssim_tuple.sort(key=lambda x:x[1],reverse=True)
                del highest_5_ssim_tuple[-1]#drop the lowest score
                assert len(highest_5_ssim_tuple) == 5, "length of the tuple have to be 5"
    assert len(highest_5_ssim_tuple) <= 5, "length of the tuple have to less or equal to 5"
    return random.choice(highest_5_ssim_tuple)[0]


def make_pairs(data_path, pairs, classes):#makes pairs of data
    # global pairs, classes, labels
    # pairs = np.array(pairs).astype("float32")
    # labels = np.array(labels).astype("float32")
    # pairs = []
    for class_ in classes:
        class_path = os.path.join(data_path, class_)
        for img_path in os.listdir(class_path):
            if np.random.uniform()<=0.25:#rescale images
                image1_path = os.path.join(class_path, img_path)
                image2_path = get_high_ssim_img_in_cls_path(img1_path=image1_path,class_path=class_path)
                scale = np.random.uniform(0.3,0.6)#scaling factor
                select_index = random.choice([1,2])
                if select_index==1:
                    s1=int(scale*IMAGE_DIMS)#scale down
                    s2 = IMAGE_DIMS
                    scale_flag=1
                else:
                    s2=int(scale*IMAGE_DIMS)#scale down
                    s1 = IMAGE_DIMS
                    scale_flag=2
                pairs+=[[image1_path, image2_path, 0, s1, s2, 1]]#smae class

                class_select = random.choice(classes)
                while class_select == class_:# keep trying if select the current class
                    class_select = random.choice(classes)
                class_path2 = os.path.join(data_path, class_select)
                image2_path = get_high_ssim_img_in_cls_path(img1_path=image1_path,class_path=class_path2)
                if scale_flag ==1:
                    s1 = IMAGE_DIMS
                    if np.random.uniform()<0.5:
                        s2=int(scale*IMAGE_DIMS)#scale down
                    else:
                        s2 = IMAGE_DIMS
                elif scale_flag ==2:
                    if np.random.uniform()<0.5:
                        select_index = random.choice([1,2])
                        if select_index==1:
                            s1=int(scale*IMAGE_DIMS)#scale down
                            s2 = IMAGE_DIMS
                        else:
                            s2=int(scale*IMAGE_DIMS)#scale down
                            s1 = IMAGE_DIMS
                scale_flag=0
                pairs+=[[image1_path, image2_path, 1, s1, s2, 1]]#different class

            image1_path = os.path.join(class_path, img_path)
            image2_path = get_high_ssim_img_in_cls_path(img1_path=image1_path,class_path=class_path)
            # image1=preprocess(image1)
            # image2=preprocess(image2)
            pairs+=[[image1_path, image2_path, 0, IMAGE_DIMS, IMAGE_DIMS, 1]]#same class


            class_select = random.choice(classes)
            while class_select == class_:# keep trying if select the current class
                class_select = random.choice(classes)
            class_path2 = os.path.join(data_path, class_select)
            image2_path = get_high_ssim_img_in_cls_path(img1_path=image1_path,class_path=class_path2)
            # image2=preprocess(image2)
            pairs+=[[image1_path, image2_path, 1, IMAGE_DIMS, IMAGE_DIMS, 1]]#different class


