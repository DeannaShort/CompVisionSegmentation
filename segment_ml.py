# This code and training data origniates from the following two sources: 
#   * https://github.com/andrewssobral/vehicle_detection_haarcascades
#   * https://www.kaggle.com/code/hamedetezadi/haar-cascade-classifier-car-detection/notebook
#
# This code is using OpenCV's Cascade Classifier with object detection following the HAAR based model

from PIL import Image
import cv2 as cv
import numpy as np

image = cv.imread('images/cars.jpg')
# Convert the image to a Numpy array
image_arr = np.array(image)

# Convert the image to grayscale
grey = cv.cvtColor(image_arr, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to the grayscale image
blur = cv.GaussianBlur(grey, (5, 5), 0)

# Apply dilation to the blurred image
dilated = cv.dilate(blur, np.ones((3, 3)))

# Apply morphological closing to the dilated image
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
closing = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)

# Use CascadeClassifier for car detection
car_cascade_src = 'cars.xml'
car_cascade = cv.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)

# Draw rectangles around each detected car and count
cnt = 0
for (x, y, w, h) in cars:
    cv.rectangle(image_arr, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cnt += 1

# Print the total number of detected cars and buses
print(cnt, " cars found")

# Convert the annotated image to PIL Image format and display it
annotated_image = Image.fromarray(image_arr)
annotated_image.show()