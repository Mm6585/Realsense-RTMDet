from mmdet.apis import inference_detector, init_detector
from mmengine.structures.instance_data import InstanceData
import os
from time import time
import numpy as np

ws_path = os.getcwd()
cur_path = os.path.abspath('.')

class RTMDet:
    def __init__(self, device='cuda:0'):
        self.config = os.path.join(cur_path, 'weights/rtmdet-ins_s.py')
        checkpoint = os.path.join(cur_path, 'weights/rtmdet-ins_s.pth')
        self.model = init_detector(self.config, checkpoint, device=device)
        self.device = device

    def get_prediction(self, image, classes=None):
        pred_start = time()
        pred_result = inference_detector(self.model, image)
        pred_time = time() - pred_start
        
        pred_labels = pred_result.pred_instances.labels.cpu().numpy()
        pred_masks = pred_result.pred_instances.masks.cpu().numpy()
        pred_scores = pred_result.pred_instances.scores.cpu().numpy()
        pred_bboxes = pred_result.pred_instances.bboxes.cpu().numpy()

        if (pred_labels.any()):
            # extract specified classes
            if (classes is not None):
                esc_strat = time()
                _class = [i in classes for i in pred_labels]
                pred_result = [pred_labels[_class], pred_masks[_class], pred_scores[_class], pred_bboxes[_class]]
                esc_time = time() - esc_strat
                return pred_result, pred_time, esc_time
            else:
                pred_result = [pred_labels, pred_masks, pred_scores, pred_bboxes]
                print(pred_result)
                return pred_result, pred_time, 0
        else:
            return None, pred_time, 0

            ### if you want to use mmdetection apis, use this. ###
            # _labels = pred_result.pred_instances.labels
            # if (self.device == 'cpu'):
            #     _class = torch.isin(_labels, torch.tensor(classes))
            # else:
            #     _class = torch.isin(_labels, torch.tensor(classes).cuda())

            # if (_class.count_nonzero() > 1):
            #     pred_result.pred_instances = InstanceData(**dict(
            #         kernels = torch.squeeze(pred_result.pred_instances.kernels[_class]),
            #         labels = torch.squeeze(pred_result.pred_instances.labels[_class]),
            #         masks = torch.squeeze(pred_result.pred_instances.masks[_class]),
            #         scores = torch.squeeze(pred_result.pred_instances.scores[_class]),
            #         bboxes = torch.squeeze(pred_result.pred_instances.bboxes[_class]),
            #         priors = torch.squeeze(pred_result.pred_instances.priors[_class])
            #     ))
            # else:
            #     pred_result = None

        # return pred_result, pred_time
    