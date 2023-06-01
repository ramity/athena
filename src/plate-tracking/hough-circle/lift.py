import sys
import pytesseract
import cv2
import numpy

capture = cv2.VideoCapture('./input/lift.mp4')

if (capture.isOpened()== False): 
  sys.exit("Error opening video stream or file")

frame_id = 0

width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

black_frame = numpy.zeros((height, width))

while(capture.isOpened()):
    ret, frame = capture.read()
    
    if ret == True:
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (11, 11))
        detected_circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1 = 50,
            param2 = 30,
            minRadius = 50,
            maxRadius = 100
        )

        # Draw circles that are detected.
        if detected_circles is not None:

            # Convert the circle parameters a, b and r to integers.
            detected_circles = numpy.uint16(numpy.around(detected_circles))

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(blur, (a, b), r, (0, 255, 0), 2)

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(blur, (a, b), 1, (0, 0, 255), 3)
                cv2.circle(black_frame, (a, b), 1, (255, 255, 255), 3)

            if frame_id % 10 == 0:
                cv2.imwrite('./output/{}.jpg'.format(frame_id), blur)
        else:
            print('No circles detected')
        
        if frame_id >= 100:
            cv2.imwrite('./output/blur.jpg'.format(frame_id), blur)
            cv2.imwrite('./output/black.jpg'.format(frame_id), black_frame)
            break

        # break
        frame_id += 1