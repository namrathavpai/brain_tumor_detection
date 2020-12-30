from sklearn.utils import shuffle

import crop_brain
import cv2
import os
import numpy as np

def load_data(dir_list):
    X = [] #image array
    y = [] #target array 1->yes 0->no
    for dir in dir_list:
        for img in os.listdir(dir):
            #preprocessing of the data
            image = crop_brain.brain_crop_resize(dir + '\\' + img)
            #normalise image
            image = cv2.normalize(image, image , 0, 255, cv2.NORM_MINMAX)
            #append image in X
            X.append(image)
            #append 0 or 1 in y
            if os.path.basename(dir) == 'yes':
                y.append([1])
            else:
                y.append([0])

    X = np.array(X)
    y = np.array(y)
    #shuffle data
    X, y = shuffle(X, y)
