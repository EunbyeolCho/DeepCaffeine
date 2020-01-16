import numpy as np

def one_hot(img):
    # print(img.shape)

    img = np.array(img)
    idx = np.argmax(img, axis = 0)

    # print(idx)

    iidx = idx.reshape(-1)
    c, w, h = img.shape
    one_hot_img = np.zeros([c,w,h])
    # print(one_hot_img.shape)

    for i, j in enumerate(iidx):
        one_hot_img[j, i//h, i%h] = 1
    # print(one_hot_img.sum())
    
    return one_hot_img
        

