from entity import VideoContainer

root = VideoContainer('./input/lift15.mp4')
root = root.load()
root.process()
root.process_corners()