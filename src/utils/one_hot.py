import numpy as np
import cv2

def one_hot(img): #img shape = c, H, W
    # print(img.shape)

    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = np.resize(img, (3,2983,2983))
    idx = np.argmax(img, axis = 0)

    # print(idx)

    iidx = idx.reshape(-1)
    c, h, w = img.shape
    one_hot_img = np.zeros([c,h,w])
    print(one_hot_img.shape)

    for i, j in enumerate(iidx):
        one_hot_img[j, i//h, i%h] = 1
        #print(i,j)
    # print(one_hot_img.sum())
    print(np.argmax(one_hot_img, axis = 0)[1332])
    #print(np.argmin(one_hot_img, axis = 2))
    return one_hot_img

