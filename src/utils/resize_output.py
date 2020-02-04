import numpy as np
import cv2

def resize_output(opt, output, img_size):
    # h,w 중 작은 사이즈에 맞춰 crop 후 resize
    H, W = np.array(img_size.squeeze()).astype(int)
    # print(H,W)

    c = np.array(opt.num_class+1).astype(int)
    output = np.array(output)
    new_output = np.zeros((c, H, W))

    if (H<1536) or (W<1536):
        print('FIND ERROR!!!! This dcm file shape(H,W) : ', H, W)
        for i in range(c):
            new_output[i, :, :] = cv2.resize(output[i, :, :], (H,W), interpolation = cv2.INTER_CUBIC)
        
    else :
        diff_h = int(round((H-1536)*0.8))
        diff_w = int(round((W-1536)*0.5))
        print('RESIZING...diff_h : {}, diff_w : {}, input_img_shape : ({},{}), output_shape : {}\n'.format(diff_h, diff_w, H, W, output.shape))
        for i in range(c):
            new_output[i, diff_h:diff_h+1536, diff_w:diff_w+1536] = cv2.resize(output[i, :, :], (1536,1536), interpolation = cv2.INTER_CUBIC)

    return new_output