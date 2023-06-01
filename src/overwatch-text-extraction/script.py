import sys
import pytesseract
import cv2
import numpy

input_file_path = "./input/overwatch-screenshot.jpg"
image = cv2.imread(input_file_path)

# compassRegion = frame[compassY1:compassY2, compassX1:compassX2]
# compassRegion = 255 - cv2.cvtColor(compassRegion, cv2.COLOR_RGB2GRAY)
# compassRegion = cv2.GaussianBlur(compassRegion, (3, 3), 0)
# ret, compassRegion = cv2.threshold(compassRegion, 128, 255, cv2.THRESH_TRUNC)
# compassRegion = cv2.erode(compassRegion, numpy.ones((2, 2), numpy.uint8), iterations=1)
# compassRegion = cv2.adaptiveThreshold(compassRegion, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
output = pytesseract.image_to_data(image, config='--psm 11')

output_file_path = "./output/text.jpg"
cv2.imwrite(output_file_path, gray)
print(output)