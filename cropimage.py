# from PIL import Image
# import numpy as np

# img = np.asarray(Image.open('lena_cv_rotate_90_clockwise.jpg'))
# img2 = np.asarray(Image.open('lena_cv_rotate_90_counterclockwise.jpg'))

# comparison = img == img2

# equal_arrays = comparison.all() 


# # print("Image 1: ", img)

# # print("Image 2: ", img2)
# if ( equal_arrays == False):
#     print("2 Image are different")
# else:
#     print("2 Image are the same")   

# # print(equal_arrays)


from PIL import Image
import numpy as np

# 510 - 0 = 510
left = 0
right = 410
# 292 - 50 = 242
top = 50
bottom = 242

img = Image.open('img1.jpg')

img_res = img.crop((left, top, right, bottom)) 
img_res.save("img1_crop2.jpg")