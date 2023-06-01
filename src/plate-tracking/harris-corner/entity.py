import numpy
import cv2
import sys
import time
import os
import pickle

class VideoContainer:

    def __init__(self, video_path):

        # Init vars if file has not been processed before
        self.video_path = video_path
        self.processed = False
        self.grayscale_frames = None
        self.corner_probabilities = None
        self.corner_masks = None
        self.edge_frames = None
        self.edge_masks = None
        self.lines = []
        self.statistics = {}

    def load(self):

        # Open video and prepare values
        capture = cv2.VideoCapture(self.video_path)

        if not capture.isOpened():
            sys.exit('Failed to open file: {}'.format(self.video_path))

        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pickle_path = './output/{}-{}-{}-{}.pkl'.format(
            self.video_path.split('/')[-1],
            self.width,
            self.height,
            self.frame_count
        )

        # Check if this file has been processed before
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as f:
                return pickle.load(f)
        return self

    def process(self):

        # Skip if loaded from pickle
        if self.processed:
            return

        # Initialize memory
        size = (self.frame_count, self.height, self.width, 3)
        self.frames = numpy.zeros(size, dtype=numpy.uint8)
        size = (self.frame_count, self.height, self.width)
        self.grayscale_frames = numpy.zeros(size, dtype=numpy.uint8)
        self.corner_probabilities = numpy.zeros(size, dtype=numpy.float32)
        self.corner_masks = numpy.zeros(size, dtype=numpy.bool_)
        self.edge_frames = numpy.zeros(size, dtype=numpy.uint8)
        self.edge_masks = numpy.zeros(size, dtype=numpy.bool_)
        self.lines = []

        # Initialize tracked statistics
        self.statistics['frame_memory_times'] = []
        self.statistics['grayscale_compute_times'] = []
        self.statistics['grayscale_memory_times'] = []
        self.statistics['corner_probabilities_compute_times'] = []
        self.statistics['corner_probabilities_memory_times'] = []
        self.statistics['corner_mask_compute_times'] = []
        self.statistics['corner_mask_memory_times'] = []
        self.statistics['edge_compute_times'] = []
        self.statistics['edge_memory_times'] = []
        self.statistics['edge_mask_compute_times'] = []
        self.statistics['edge_mask_memory_times'] = []
        self.statistics['line_compute_times'] = []
        self.statistics['line_memory_times'] = []

        frame_offset = 0
        capture = cv2.VideoCapture(self.video_path)

        while (capture.isOpened()):

            ret, frame = capture.read()

            if ret == True:

                print('Processing frame {} of {}'.format(frame_offset, self.frame_count-1), end='\r')

                # Store frame
                start = time.process_time()
                self.frames[frame_offset] = frame
                end = time.process_time()
                delta = end - start
                self.statistics['frame_memory_times'].append(delta)

                # Compute grayscale
                start = time.process_time()
                grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                end = time.process_time()
                delta = end - start
                self.statistics['grayscale_compute_times'].append(delta)

                # Store grayscale
                start = time.process_time()
                self.grayscale_frames[frame_offset] = grayscale
                end = time.process_time()
                delta = end - start
                self.statistics['grayscale_memory_times'].append(delta)

                # Compute corner probabilities
                start = time.process_time()
                corner_probabilities = cv2.cornerHarris(grayscale, 2, 3, 0.04)
                end = time.process_time()
                delta = end - start
                self.statistics['corner_probabilities_compute_times'].append(delta)

                # Store corner probabilities
                start = time.process_time()
                self.corner_probabilities[frame_offset] = corner_probabilities
                end = time.process_time()
                delta = end - start
                self.statistics['corner_probabilities_memory_times'].append(delta)

                # Compute corner mask
                start = time.process_time()
                corner_criteria = 0.01 * corner_probabilities.max()
                corner_mask = corner_probabilities > corner_criteria
                end = time.process_time()
                delta = end - start
                self.statistics['corner_mask_compute_times'].append(delta)

                # Store corner mask
                start = time.process_time()
                self.corner_masks[frame_offset] = corner_mask
                end = time.process_time()
                delta = end - start
                self.statistics['corner_mask_memory_times'].append(delta)


                # Compute edge frame
                start = time.process_time()
                edge_frame = cv2.Canny(grayscale, 100, 200)
                end = time.process_time()
                delta = end - start
                self.statistics['edge_compute_times'].append(delta)

                # Store edge frame
                start = time.process_time()
                self.edge_frames[frame_offset] = edge_frame
                end = time.process_time()
                delta = end - start
                self.statistics['edge_memory_times'].append(delta)

                # Compute edge mask
                start = time.process_time()
                edge_mask = edge_frame.astype(numpy.bool_)
                end = time.process_time()
                delta = end - start
                self.statistics['edge_mask_compute_times'].append(delta)

                # Store edge mask
                start = time.process_time()
                self.edge_masks[frame_offset] = edge_mask
                end = time.process_time()
                delta = end - start
                self.statistics['edge_mask_memory_times'].append(delta)

                # Compute lines
                start = time.process_time()
                linesP = cv2.HoughLinesP(edge_frame, 1, numpy.pi / 180, 50, None, 50, 10)
                end = time.process_time()
                delta = end - start
                self.statistics['line_compute_times'].append(delta)

                # Store lines
                start = time.process_time()
                self.lines.append(linesP)
                end = time.process_time()
                delta = end - start
                self.statistics['line_memory_times'].append(delta)

            else:
                break

            frame_offset += 1

        capture.release()

        # Output to pickle file to skip processing next run
        self.processed = True
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self, f)
        print('')

    def print_statistics(self):

        for key in self.statistics.keys():
            print(key, self.statistics[key])

    def print_compute_time(self):

        sum = 0
        for key in self.statistics.keys():
            if 'compute' in key:
                for z in self.statistics[key]:
                    sum += z
        print('compute_time', sum)
    
    def print_memory_time(self):

        sum = 0
        for key in self.statistics.keys():
            if 'memory' in key:
                for z in self.statistics[key]:
                    sum += z
        print('memory_time', sum)

    def process_corners(self):

        global_corners = numpy.zeros((self.height, self.width), dtype=numpy.uint16)
        unique_corners = numpy.zeros((self.height, self.width), dtype=numpy.uint8)

        for frame_offset in range(self.frame_count):

            print('Adding corners from frame {} of {}'.format(frame_offset, self.frame_count-1), end='\r')

            # Update global corner counts
            global_corners += self.corner_masks[frame_offset]

            # Add points that aren't in previous frame
            if frame_offset > 0:
                unique_corners += numpy.logical_xor(
                    self.corner_masks[frame_offset - 1],
                    self.corner_masks[frame_offset]
                )

        print('')
        unique_corners_mask = unique_corners == 1
        global_corners_mask = global_corners == 1

        cv2.imwrite('./output/global_corners.jpg', global_corners)
        cv2.imwrite('./output/unique_corners.jpg', (unique_corners / unique_corners.max()) * 255)
        cv2.imwrite('./output/global_corners_mask.jpg', global_corners_mask * 255)
        cv2.imwrite('./output/unique_corner_mask.jpg', unique_corners_mask * 255)

        # Building pip installed opencv-contrib-python-headless
        # mp4v - works but not supported by browsers
        # drac - works but not supported by anything
        # vp09 - works but slow writing time

        frame_video_out = cv2.VideoWriter(
            './output/frame_video.mp4',
            cv2.VideoWriter_fourcc(*'avc1'),
            30,
            (self.width, self.height),
            3
        )

        mask_video_out = cv2.VideoWriter(
            './output/mask_video.mp4',
            cv2.VideoWriter_fourcc(*'avc1'),
            30,
            (self.width, self.height),
            3
        )
        combined_video_out = cv2.VideoWriter(
            './output/combined_video.mp4',
            cv2.VideoWriter_fourcc(*'avc1'),
            30,
            (self.width * 2, self.height),
            3
        )


        for frame_offset in range(self.frame_count):

            print('Outputting video frame {} of {}'.format(frame_offset, self.frame_count-1), end='\r')

            black = numpy.zeros((self.height, self.width, 3), dtype=numpy.uint8)
            frame = self.frames[frame_offset]
            combined_mask = numpy.logical_and(
                self.corner_masks[frame_offset],
                global_corners_mask
            )

            # convert mask to image, erode, convert back to mask
            combined_mask = combined_mask.astype(numpy.uint8)
            combined_mask = combined_mask * 255
            kernel = numpy.ones((5, 5), numpy.uint8)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations = 1)
            combined_mask = combined_mask.astype(numpy.bool_)

            black[global_corners_mask] = [128, 128, 128]
            black[combined_mask] = [255, 255, 255]
            frame[combined_mask] = [0, 255, 0]

            combined_frame = numpy.concatenate((black, frame), axis=1)

            mask_video_out.write(black)
            frame_video_out.write(frame)
            combined_video_out.write(combined_frame)

        print('')
        frame_video_out.release()
        mask_video_out.release()
        combined_video_out.release()

