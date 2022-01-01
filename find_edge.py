import os, shutil
import cv2
import numpy as np

def resize(path_to_img, scale = 60):
    img = cv2.imread(path_to_img, cv2.IMREAD_UNCHANGED)

    scale_percent = scale # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def is_crop(path_to_img, threshold=0.1, min_width= 10, min_height=10, max_width=300, max_height=300):
    img = cv2.imread(path_to_img, 0).T

    w, h = img.shape

    sens = threshold  # (0-1]
    meanofimg = np.mean(img)*sens
    dataw = [w, 0]
    datah = [h, 0]
    
    tempw = [w, 0]
    temph = [h, 0]
    for i in range(w):
        if np.mean(img[i]) > meanofimg:
            if i < dataw[0]:
                dataw[0] = i
            else:
                dataw[1] = i
                tempw[0] = i
    img = img.T
    meanofimg = np.mean(img)*sens
    
    for i in range(h):
        if np.mean(img[i]) > meanofimg:
            if i < datah[0]:
                datah[0] = i
            else:
                datah[1] = i
                temph[0] = i

    left = dataw[0]
    right = tempw[0]

    top = datah[0]
    bottom = temph[0]

    print("Height start point: ", top)
    print("Height end point   :", bottom)  

    print("\nWidth start point: ", left)
    print("Width end point  :", right)


    img = img[datah[0]:datah[1], dataw[0]:dataw[1]]
    # cv2.imwrite("output.jpg", img)
    h2, w2 = img.shape

    h_img = h - h2
    w_img = w - w2
    print("\nOrigin size: ", h, w)
    # print("Crop detect size: ", h2, w2)
    # # print(int(829/2))
    # # print(h_img, w_img)
    # if ( h_img >= max_height or w_img >= max_width):
    #     return True
    # else:    
    #     if (h_img <= min_height and w_img <= min_width):
    #         return True
    #     else:
    #         return False

    if (
        (top <= 1 and h - bottom <= 1 and left <= 1 and w - right <= 1) or # image is normal
        
        (top > 1 and h - bottom <= 1  and left <= 1 and w - right <= 1) or # top crop only
        (top <= 1 and h - bottom > 1  and left <= 1 and w - right <= 1) or # bottom crop only

        (top <= 1 and h - bottom <= 1  and left > 1 and w - right <= 1) or # left crop only
        (top <= 1 and h - bottom <= 1  and left <= 1 and w - right > 1) or # right crop only
        
        (top >= 50 and h - bottom >= 50 or left >= 50 and w- right >= 50) 
    ):
        return True # image is normal
    elif (
         (top > 1 and h - bottom > 1) or
         (left > 1 and w - right > 1)
    ):
        return False   # image is crop


# flag = is_crop("image/93.jpg")
# if (flag == True):
#     print("Image is normal")
# else:
#     print("Image is crop")


# test 
file_list = os.listdir("image")

# create folder if not exist
if not os.path.exists('normal'):
    os.makedirs('normal')
if not os.path.exists('crop'):
    os.makedirs('crop')
    
# read images from folder    
for i in range(0, len(file_list)):
    path = 'image/' + str(i+1) + '.jpg'
    # image = cv2.imread(path, 0 ).T
    # h, w = image.shape
    # if ( h >= 500 or w >= 500):
    #     cv2.imwrite(path, resize(path))

    state = is_crop(path)

    if (state == True):
        shutil.copy(path, 'normal')
    else:
        shutil.copy(path, 'crop')

# note: resize to (1920, 720)