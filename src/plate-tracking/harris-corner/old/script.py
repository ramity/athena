import sys
import pytesseract
import cv2
import numpy
from matplotlib import pyplot as plt

video = cv2.VideoCapture('./input/lift3.mp4')

if (video.isOpened()== False): 
  sys.exit("Error opening video stream or file")

frame_id = 0

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

global_black_frame = numpy.zeros((height, width), dtype=numpy.uint8)

while(video.isOpened()):
    ret, frame = video.read()

    if ret == True:

        local_black_frame = global_black_frame.copy()
        local_black_lines = global_black_frame.copy()
        local_black_combined = global_black_frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        local_black_frame[dst > 0.01 * dst.max()] = 255
        local_black_combined[dst > 0.01 * dst.max()] = 255
        global_black_frame[dst > 0.01 * dst.max()] = 255

        edges = cv2.Canny(gray, 100, 200)
        
        linesP = cv2.HoughLinesP(edges, 1, numpy.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(local_black_lines, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)
                cv2.line(local_black_combined, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)

        if frame_id % 10 == 0:
            local_black_frame = numpy.concatenate((local_black_frame, local_black_lines, local_black_combined, gray), axis=1)
            cv2.imwrite('./output/{}.jpg'.format(frame_id), local_black_frame)
    else:
        break

    frame_id += 1

cv2.imwrite('./output/black.jpg'.format(frame_id), global_black_frame)

# Calculate x and y histograms from global_black_frame
x_bins = numpy.zeros(width)
y_bins = numpy.zeros(height)
for x in range(width):
    for y in range(height):
        if global_black_frame[y][x] == 255:
            x_bins[x] += 1
            y_bins[y] += 1

print(x_bins)
print(y_bins)

plt.bar(range(width), x_bins)
plt.suptitle('X bins')
plt.xlabel('Pixel offset')
plt.ylabel('Count')
plt.savefig('./output/x_bins.jpg')
plt.clf()

plt.barh(range(height-1, -1, -1), y_bins)
plt.suptitle('Y bins')
plt.xlabel('Count')
plt.ylabel('Pixel offset')
plt.savefig('./output/y_bins.jpg')
plt.clf()