import image
import cv2
import os

import pipeline as pl
from rtmdet import RTMDet

device = 'cpu'
pipeline, _ = pl.create_pipeline()
model = RTMDet(device)

ws_path = os.getcwd()
track_path = os.path.join(ws_path, 'results/masked_color_image')
os.makedirs(track_path, exist_ok=True)
index = len(os.listdir(track_path)) + 1
save_path = os.path.join(track_path, 'result' + str(index))
os.makedirs(save_path, exist_ok=True)

frame_size = (640, 480)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30.0
file_path = os.path.join(save_path, 'result.avi')
out = cv2.VideoWriter(file_path, fourcc, fps, frame_size)

classes = [0]

try:
    elapsed_time = 0
    i = 1
    while (True):

        color_image = image.get_images(pipeline, 1)

        # get detection result
        result, pred_time = model.get_prediction(color_image, classes)

        if (result == None):
            print('frame', i, ': prediction time =', round(pred_time, 2), 's', '\tobject not found for the specified classes')

            cv2.imshow('masked image', color_image)

            out.write(color_image)
            elapsed_time += pred_time

        else:
            masked_image = image.get_masked_image(color_image, result)

            print('frame', i, ': prediction time =', round(pred_time, 2), 's')
            cv2.imshow('masked image', masked_image)

            # save video
            out.write(masked_image)
            elapsed_time += pred_time

        if (cv2.waitKey(1) == 27):
            break

        i += 1

finally:
    elapsed_time = round(elapsed_time, 2)
    print('-----------------------------------------')
    print('job finished')
    print('total elapsed time =', elapsed_time)
    print('average elapsed time per frame =', round(elapsed_time/i, 2))
    print('save masked video', file_path)
    print('-----------------------------------------')
    pl.stop_pipeline(pipeline)
    out.release()
    cv2.destroyAllWindows()