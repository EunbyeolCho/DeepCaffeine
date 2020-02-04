import numpy as np
import cv2

def resize_output(opt, output, img_size):
    # h,w 중 작은 사이즈에 맞춰 crop 후 resize
    # print(img_size)
    H, W = np.array(img_size.squeeze()).astype(int)
    # print(H,W)
    c = np.array(opt.num_class+1).astype(int)
    output = np.array(output)
    new_output = np.zeros((c, H, W))

    if (H<2048) or (W<2048):
        for i in range(c):
            new_output[i, :, :] = cv2.resize(output[i, :, :], (H,W), interpolation = cv2.INTER_CUBIC)
    else :
        diff_h = int(round((H-2048)*0.8))
        diff_w = int(round((W-2048)*0.5))
        # print(type(diff_h))
        # print(output.shape)
        for i in range(c):
            new_output[i, diff_h:diff_h+2048, diff_w:diff_w+2048] = cv2.resize(output[i, :, :], (2048,2048), interpolation = cv2.INTER_CUBIC)

    return new_output