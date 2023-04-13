from rtmdet import RTMDet
import pipeline as pl
import image
import os
import cv2
import numpy as np
import argparse
from time import time

def main(opt):
    device = opt.device
    ws_path = os.getcwd()
    track_path = os.path.join(ws_path, 'results/track')
    os.makedirs(track_path, exist_ok=True)
    index = len(os.listdir(track_path)) + 1
    save_path = os.path.join(track_path, 'result' + str(index))
    os.makedirs(save_path, exist_ok=True)

    model = RTMDet()
    pipeline = pl.create_pipeline()

    frame_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0
    file_path = os.path.join(save_path, 'result.avi')
    out = cv2.VideoWriter(file_path, fourcc, fps, frame_size)

    elapsed_time = 0
    i = 0
    try:
        while (True):
            color_image, _ = image.get_images(pipeline)

            # get detection result
            result, pred_time, esc_time = model.get_prediction(color_image, [0], device=device)
            masked_image = model.get_masked_image(color_image, result)

            print('frame', i, ': prediction time =', round(pred_time, 2), 's\textract specified classes time =', round(esc_time, 2), 's')
            cv2.imshow('masked image', masked_image)

            # save video
            out.write(masked_image)
            elapsed_time += pred_time + esc_time

            if (cv2.waitKey(1) == 27):
                break

            i += 1

    finally:
        elapsed_time = round(elapsed_time, 2)
        print('-----------------------------------------')
        print('job finished')
        print('total elapsed time =', elapsed_time)
        print('average elapsed time per frame =', round(elapsed_time/i, 2))
        print('-----------------------------------------')
        pl.stop_pipeline(pipeline)
        out.release()
        cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu or etc..')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
    
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)