import pipeline as pl
import numpy as np
import cv2
import pyrealsense2 as rs
import os
from mmdet.registry import VISUALIZERS
from cfg import get_classes, get_colors

model_classes = get_classes()
colors = get_colors()

def get_images(pipeline, type=0):
    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())
    if (type == 1):
        return color_image
    
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    if (type == 2):
        return depth_image
        # return depth_colormap

    return color_image, depth_colormap

# def get_masked_image(model, color_image, result):

#     visualizer = VISUALIZERS.build(model.model.cfg.visualizer)
#     visualizer.dataset_meta = model.model.dataset_meta
#     visualizer.add_datasample(
#         'result',
#         color_image,
#         data_sample=result,
#         draw_gt=False,
#         wait_time=0,
#     )
#     return visualizer.get_image()

def get_masked_image(color_image, result, threshold=0.5):
    _class, mask, score, bbox = result[0], result[1], result[2], result[3]

    mask_list, class_list, score_list, bbox_list = [], [], [], []
    # extract masks, classes, scores, bboxes
    for i in range(len(_class)):
        if (score[i] > threshold):
            left = bbox[i][0]
            top = bbox[i][1]
            right = bbox[i][2]
            bottom = bbox[i][3]

            w = int(right - left)
            h = int(bottom - top)
            p1 = (int(left), int(top))
            p2 = (int(left+w), int(top+h))
            
            mask_image = np.zeros(color_image.shape, dtype=np.uint8)
            mask_image[mask[i] == True] = colors[_class[i]]
            mask_list.append(mask_image)
            class_list.append(_class[i])
            score_list.append(round(score[i], 2))
            bbox_list.append([p1, p2])
    mask_image = mask_list[0]
    # merge masks and color image
    if (len(mask_list) > 1):
        for i in range(len(mask_list)):
            mask_image = cv2.bitwise_or(mask_image, mask_list[i])
    masked_image = cv2.addWeighted(color_image, 0.8, mask_image, 0.2, 0)
    # draw bbox & put class, score text
    for i in range(len(mask_list)):
        p1 = bbox_list[i][0]
        p2 = bbox_list[i][1]
        cv2.rectangle(masked_image, p1, p2, colors[class_list[i]], 2, 1)
        cv2.putText(masked_image, 'class=' + model_classes[class_list[i]] + 
                    ' score='+str(score_list[i]), (p1[0], p1[1]-10), 2, 1, colors[class_list[i]], 2)

    return masked_image

def save_masked_vid(model, pipeline, classes=None):
    ws_path = os.getcwd()
    track_path = os.path.join(ws_path, 'results/track')
    os.makedirs(track_path, exist_ok=True)
    index = len(os.listdir(track_path)) + 1
    save_path = os.path.join(track_path, 'result' + str(index))
    os.makedirs(save_path, exist_ok=True)

    frame_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0
    file_path = os.path.join(save_path, 'result.avi')
    out = cv2.VideoWriter(file_path, fourcc, fps, frame_size)

    elapsed_time = 0
    i = 0
    try:
        while (True):
            if (i == 0):
                i += 1
                continue

            color_image = get_images(pipeline, 1)

            # get detection result
            result, pred_time, esc_time = model.get_prediction(color_image, classes)

            if (result == None):
                print('frame', i, ': prediction time =', round(pred_time, 2), 's', '\tobject not found for the specified classes')

                cv2.imshow('masked image', color_image)

                out.write(color_image)
                elapsed_time += pred_time

            else:
                masked_image = get_masked_image(color_image, result)

                print('frame', i, ': prediction time =', round(pred_time, 2), 's\textract specified classes time =', round(esc_time, 2), 's')
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