import cv2
def resize(path_to_img):
    img = cv2.imread(path_to_img, cv2.IMREAD_UNCHANGED)

    # scale_percent = scale  # percent of original size
    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    width = 224
    height = 224
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

img = resize('./test3.jpg')
cv2.imwrite("test4.jpg", img)