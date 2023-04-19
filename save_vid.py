import cv2
import os
import image
import pipeline as pl

ws_path = os.getcwd()
color_path = os.path.join(ws_path, 'color.avi')
depth_path = os.path.join(ws_path, 'depth.avi')

frame_size = (640, 480)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30.0
color_out = cv2.VideoWriter(color_path, fourcc, fps, frame_size)
depth_out = cv2.VideoWriter(depth_path, fourcc, fps, frame_size)

pipeline, profile = pl.create_pipeline()


while (True):
    color_image, depth_image = image.get_images(pipeline)
    color_out.write(color_image)
    depth_out.write(depth_image)

    cv2.imshow('color', color_image)

    if(cv2.waitKey(1) == 27 & 0xFF):
        break

pl.stop_pipeline(pipeline)