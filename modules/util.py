import numpy as np
import cv2
import random
import os

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


def make_pairs(data_path, pairs, classes):#makes pairs of data
    # global pairs, classes, labels
    # pairs = np.array(pairs).astype("float32")
    # labels = np.array(labels).astype("float32")
    # pairs = []

    random.seed(15)#fixed random state 
    np.random.seed(15)

    for class_ in classes:
        class_path = os.path.join(data_path, class_)
        for img_path in os.listdir(class_path):
            if np.random.uniform()<=0.25:#rescale images
                image1 = os.path.join(class_path, img_path)
                image_select=random.choice(os.listdir(class_path))
                image2 = os.path.join(class_path, image_select)
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
                pairs+=[[image1, image2, 0, s1, s2,1]]#smae class

                class_select = random.choice(classes)
                while class_select == class_:# keep trying if select the current class
                    class_select = random.choice(classes)
                class_path2 = os.path.join(data_path, class_select)
                image_select=random.choice(os.listdir(class_path2))
                image2 = os.path.join(class_path2, image_select)
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
                pairs+=[[image1, image2, 1, s1, s2,1]]#different class

            image1 = os.path.join(class_path, img_path)
            image_select=random.choice(os.listdir(class_path))
            image2 = os.path.join(class_path, image_select)
            # image1=preprocess(image1)
            # image2=preprocess(image2)
            pairs+=[[image1, image2, 0, IMAGE_DIMS, IMAGE_DIMS,1]]#same class


            class_select = random.choice(classes)
            while class_select == class_:# keep trying if select the current class
                class_select = random.choice(classes)
            class_path2 = os.path.join(data_path, class_select)
            image_select=random.choice(os.listdir(class_path2))
            image2 = os.path.join(class_path2, image_select)
            # image2=preprocess(image2)
            pairs+=[[image1, image2, 1, IMAGE_DIMS, IMAGE_DIMS,1]]#different class


