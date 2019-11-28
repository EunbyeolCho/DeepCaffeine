import cv2
import os

def load_ben_color(path, sigmaX=30):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = crop_image_from_gray(image)
    # image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

    return image

path = "../../data/eye-data/train/AMD"
c_path = "../../data/eye-data/train/AMD_c30"

img_list = os.listdir(path)
img_len = len(img_list)

if not os.path.exists(c_path):
        os.makedirs(c_path)


print("num of image : ",img_len)
for i, f in enumerate(img_list) : 
    c_img = load_ben_color(os.path.join(path, f))
    cv2.imwrite(os.path.join(c_path,f[:-4]+"_c.jpg"), c_img)
    print(i)

