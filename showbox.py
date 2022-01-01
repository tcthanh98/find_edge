from cv2 import cv2  
import matplotlib.pyplot as plt
# path  
path = './image/68.jpg'

image = cv2.imread(path)   
window_name = 'Image'
  
# represents the top left corner of rectangle 
start_point = (0, 10)  # (width, height)
  
# represents the bottom right corner of rectangle 
end_point = (224, 148) 
  
color = (255, 0, 0) 
thickness = 1
image = cv2.rectangle(image, start_point, end_point, color, thickness) 

# cv2.imshow(window_name, image)
# cv2.waitKey(0)
plt.imshow(image)
plt.show()