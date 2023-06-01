import sys
import pytesseract
import cv2
import numpy

filename = "IKz1wgqzNdI.mp4"
filepath = "/var/raw/{0}".format(filename)
capture = cv2.VideoCapture(filepath)

print(filepath)

if not capture.isOpened():
    print("Could not open video: {0}".format(filepath))
    sys.exit(1)

while(capture.isOpened()):

    ret, frame = capture.read()

    if not ret:
        print("End of video capture")
        break

    compassX1 = 935
    compassX2 = 985
    compassY1 = 91
    compassY2 = 118

    compassRegion = frame[compassY1:compassY2, compassX1:compassX2]
    compassRegion = 255 - cv2.cvtColor(compassRegion, cv2.COLOR_RGB2GRAY)
    compassRegion = cv2.GaussianBlur(compassRegion, (3, 3), 0)
    ret, compassRegion = cv2.threshold(compassRegion, 128, 255, cv2.THRESH_TRUNC)
    # compassRegion = cv2.erode(compassRegion, numpy.ones((2, 2), numpy.uint8), iterations=1)
    # compassRegion = cv2.adaptiveThreshold(compassRegion, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    output = pytesseract.image_to_string(compassRegion, config="--psm 7 nobatch digits -c tessedit_char_whitelist=1234567890 -c tessedit_char_blacklist=.")

    cv2.imwrite("test.png", compassRegion)
    print(output)

    