import numpy as np
import cv2

def resize_output(opt, output, img_size):
    # h,w 중 작은 사이즈에 맞춰 crop 후 resize
    H, W = np.array(img_size.squeeze()).astype(int)
    c = np.array(opt.num_class+1).astype(int)
    output = np.array(output)
    new_output = np.zeros((c, H, W))
 
    if(H > W):
        diff = H-W
        for i in range(c):
            new_output[i, diff:H, 0:W] = cv2.resize(output[i, :, :], (W, W), interpolation=cv2.INTER_CUBIC)
    else:
        diff = W-H
        for i in range(c):
            new_output[i, 0:H, diff:W] = cv2.resize(output[i, :, :], (H, H), interpolation=cv2.INTER_CUBIC)

    return new_output