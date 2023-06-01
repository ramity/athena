import numpy
import cv2
import sys

class VideoContainer:

    def __init__(self):

        self.video_path = ''
        self.video_capture = None
        self.video_width = 0
        self.video_height = 0
        self.video_frame_count = 0

        self.video_grays = None

        self.video_corners = []
        self.video_edges = []
        self.video_lines = []

    def process(self):

        self.video_capture = cv2.VideoCapture(self.video_path)
        self.video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_grays = numpy.zeros((
            self.video_frame_count,
            self.video_height,
            self.video_width
        ), dtype=numpy.uint8)

        frame_offset = 0

        while (self.video_capture.isOpened()):

            print('Frame {} of {}'.format(frame_offset, self.video_frame_count), end='\r')
            ret, frame = self.video_capture.read()

            if ret == True:

                # Add gray
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.video_grays[frame_offset] = gray

                # Add corners
                dst = cv2.cornerHarris(gray, 2, 3, 0.04)
                output = dst > 0.01 * dst.max()
                print()
                print(output[0])
                sys.exit(0)
                self.video_corners.append(dst > 0.01 * dst.max())

                # Add edges
                edges = cv2.Canny(gray, 100, 200)
                self.video_edges.append(edges == 255)

                # Add lines
                linesP = cv2.HoughLinesP(edges, 1, numpy.pi / 180, 50, None, 50, 10)
                self.video_lines.append(linesP)

            else:
                break

            frame_offset += 1

    def process_corners(self):

        for frame_offset in range(self.video_frame_count):

            # y, x format

            print(self.video_corners[frame_offset])
            print(self.video_corners[frame_offset].shape)
            break

