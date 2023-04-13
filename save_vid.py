import cv2
import image
import pipeline as pl

if __name__ == "__main__":
    color_vid_name = input('Input Color Video Name : ')
    depth_vid_name = input('Input Depth Video Name : ')

    frame_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0
    color_writer = cv2.VideoWriter(color_vid_name+'.avi', fourcc, fps, frame_size)
    depth_writer = cv2.VideoWriter(depth_vid_name+'.avi', fourcc, fps, frame_size)
    
    pipeline = pl.create_pipeline()

    try:
        i = 0
        while (i < 1000):
            color_image, depth_image = image.get_images(pipeline)  # images[0=color_image, 1=depth_image], size=(480, 640, 3)
            color_writer.write(color_image)
            depth_writer.write(color_image)
            i += 1
            
    finally:
        pl.stop_pipeline(pipeline)
        color_writer.release()
        depth_writer.release()