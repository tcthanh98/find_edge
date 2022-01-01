from PIL import Image
import random as rng
import time
from cv2 import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def is_crop_img(src):
    img = cv.imread(src)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # threshold
    t = 127
    thresh = cv.threshold(gray, 0, t, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    # get bounds of white pixels
    white = np.where(thresh == t)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(
        white[0]), np.max(white[1]), np.max(white[0])

    crop = gray[ymin:ymax, xmin:xmax]
    hh, ww = crop.shape

    # do adaptive thresholding
    thresh2 = cv.adaptiveThreshold(
        crop, t, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 1.1)

    # apply morphology
    kernel = np.ones((1, 7), np.uint8)
    morph = cv.morphologyEx(thresh2, cv.MORPH_CLOSE, kernel)

    kernel = np.ones((5, 5), np.uint8)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel)

    # invert
    morph = t - morph

    # get contours (presumably just one) and its bounding box
    contours = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x, y, w, h = cv.boundingRect(cntr)
        print(x, y, w, h)

    # draw bounding box on input
    bbox = img.copy()
    cv.rectangle(bbox, (x+xmin, y+ymin), (x+xmin+w, y+ymin+h), (0, 0, 255), 1)

    cv.imshow('img_thresh.png', thresh)
    cv.imshow('img_crop.png', crop)
    cv.imshow('img_bbox.png', bbox)
    cv.waitKey(0)

    # test if contour touches sides of image
    if x == 0 or y == 0 or x+w == ww or y+h == hh:
        return False
        # print('region touches the sides:\n=> Image is not crop')
    else:
        return True
        # print('region does not touch the sides:\n=> Image is crop')


def is_grey_scale(img_path):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r != g != b:
                return False
    return True


def is_black_border(path, th_w=3, th_h=3):
    print('path:', path)
    t0 = time.time()
    dim = (224, 224)
    im = cv.imread(path)
    # img_resized = cv.resize(im, dim)
    # im = cv.resize(im, dim, interpolation = cv.INTER_AREA)
    im = cv.resize(im, dim)
    # print(im.shape)
    t = 5
    if is_grey_scale(path):
        print('the image is gray')
        t = 20 # 68
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    thresh = cv.threshold(imgray, t, 20, cv.THRESH_TOZERO)[1]
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        # print(contours_poly[i])
        boundRect[i] = cv.boundingRect(contours_poly[i])

    img_h, img_w = imgray.shape
    bbox = []
    temp_dim = []

    for i, con in enumerate(boundRect):
        _, _, w, h = con
        if i == 0:
            temp_dim.append((w, h))
            bbox.append(con)
        else:
            if temp_dim[0] < (w, h):
                temp_dim[0] = (w, h)
                bbox[0] = con

    # print('original shape:', img_w, img_h)
    # print('bbox shape:', bbox[0][2], bbox[0][3])
    # print('bbox:', bbox)
    # print(contours_poly[1])

    # Show figure
    (x, y) = (100, 100)  # min
    (width, height) = (0, 0)  # max
    x = 100
    y = 100
    width = 0
    height = 0
    for i in range(len(contours)):

        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        if ( cv.contourArea(contours_poly[i]) < 10 ):
            cv.drawContours(thresh, [c], -1, (0,0,0), -1)
        else:        
            cv.drawContours(im, contours_poly, i, color)
            cv.rectangle(im, (int(boundRect[i][0]), int(boundRect[i][1])),
                        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 1)

            # print( int(boundRect[i][0]), int(boundRect[i][1]),
            #             int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3]))

            # if ((x, y) > (int(boundRect[i][0]), int(boundRect[i][1]))):
            #     (x, y) = (int(boundRect[i][0]), int(boundRect[i][1]))

            # if ((width, height) < (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3]))):
            #     (width, height) = (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3]))  

            if ( x > int(boundRect[i][0])):
                x = int(boundRect[i][0])

            if ( y > int(boundRect[i][1])):
                y = int(boundRect[i][1])           

            if ( width < int(boundRect[i][0]+boundRect[i][2])):
                width = int(boundRect[i][0]+boundRect[i][2])
    
            if ( height < int(boundRect[i][1]+boundRect[i][3])):
                height = int(boundRect[i][1]+boundRect[i][3])

    print((x, y), (width, height))

    # cv.imshow('figure', im)
    # cv.waitKey(0)
    start_point = (x, y)
    end_point = (width, height)

    color = (255, 0, 0)
    thickness = 1
    new_img = cv.rectangle(im, start_point, end_point, color, thickness)

    # plt.imshow(im)

    original_area = img_w * img_h  # dien tich anh ban dau
    content_area = (width - x) * (height - y)  # dien tich bbox


    left_area = x * img_h
    right_area = (img_w - width) * img_h

    top_area = (img_w - x - (img_w - width)) * y
    bottom_area = (img_w - x - (img_w - width)) * (img_h - height)

    # percent of black border in left and right of image
    width_border_percent= int((left_area + right_area) * 100 / original_area)
    # percent of black border on top and bottom of image
    height_border_percent = int((top_area + bottom_area) * 100 / original_area)

    print("Width border(%): ", width_border_percent)
    print("height border(%): ", height_border_percent)  
    print(f'{time.time() - t0:.2f}s')


    

    # h1, w1, h2, w2 = bbox[0]

    # original_area = img_w * img_h  # dien tich anh ban dau
    # content_area = bbox[0][2] * bbox[0][3]  # dien tich bbox

    # width_img = dim[0]
    # height_img = dim[1]

    # left_area = w1 * height_img
    # right_area = (width_img - w2 - w1) * height_img

    # top_area = w2 * h1
    # bottom_area = w2 * (height_img - h1 - h2)

    # # percent of black border in left and right of image
    # height_border_percent = int((left_area + right_area) * 100 / original_area)
    # # percent of black border on top and bottom of image
    # width_border_percent = int((top_area + bottom_area) * 100 / original_area)

    # print("Width border percent: ", width_border_percent)
    # print("height border percent: ", height_border_percent)
  
    # if (
    #     # (left_area != 0 and right_area == 0 and top_area == 0 and bottom_area == 0) or
    #     # (left_area == 0 and right_area != 0 and top_area == 0 and bottom_area == 0) or
    #     # (left_area == 0 and right_area == 0 and top_area != 0 and bottom_area == 0) or
    #     # (left_area == 0 and right_area == 0 and top_area == 0 and bottom_area != 0) or 
    #     ( (width_border_percent + height_border_percent) >= 80 ) 
    #     ):
    #     return False
    # else:

    if width_border_percent >= th_w or height_border_percent >= th_h:
        print("True")
        plt.title(path)
        plt.imshow(new_img)
        plt.show()        
        return True
    else:
        print("False")
        plt.title(path)
        plt.imshow(new_img)
        plt.show()  
        return False

if __name__ == "__main__":
    # t = 2 => 48

    print(is_black_border('./image/1613985837.png'))  # 71
    # print(is_black_border('test4.jpg'))
    # print(is_black_border('/home/vienph/Downloads/crop_image/normal/15.jpg'))

    # file_list = os.listdir("image")
    # for i in range(0, len(file_list)):
    #     path = 'image/' + str(i+1) + '.jpg'
    #     print(is_black_border(path))
