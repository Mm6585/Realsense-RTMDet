from mmdet.apis import inference_detector, init_detector
from mmengine.structures.instance_data import InstanceData
import os
from mmdet.registry import VISUALIZERS
from time import time
import torch

ws_path = os.getcwd()
cur_path = os.path.abspath('.')

class RTMDet:
    def __init__(self, device='cuda:0'):
        config = os.path.join(cur_path, 'weights/rtmdet-ins_s.py')
        checkpoint = os.path.join(cur_path, 'weights/rtmdet-ins_s.pth')
        self.model = init_detector(config, checkpoint, device=device)
        self.device = device

    def get_prediction(self, image, classes=None):
        pred_start = time()
        pred_result = inference_detector(self.model, image)
        pred_time = time() - pred_start
        
        # extract specified classes
        if (classes is not None):
            # keys = ['kernels', 'labels', 'masks', 'scores', 'bboxes', 'priors']

            _labels = pred_result.pred_instances.labels
            if (self.device == 'cpu'):
                _class = torch.isin(_labels, torch.tensor(classes))
            else:
                _class = torch.isin(_labels, torch.tensor(classes).cuda())

            if(_class.any()):
                pred_result.pred_instances = InstanceData(**dict(
                    kernels = torch.squeeze(pred_result.pred_instances.kernels[_class]),
                    labels = torch.squeeze(pred_result.pred_instances.labels[_class]),
                    masks = torch.squeeze(pred_result.pred_instances.masks[_class]),
                    scores = torch.squeeze(pred_result.pred_instances.scores[_class]),
                    bboxes = torch.squeeze(pred_result.pred_instances.bboxes[_class]),
                    priors = torch.squeeze(pred_result.pred_instances.priors[_class])
                ))
            else:
                pred_result = None

        return pred_result, pred_time
    
    def get_masked_image(self, color_image, result):

        visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
        visualizer.dataset_meta = self.model.dataset_meta
        visualizer.add_datasample(
            'result',
            color_image,
            data_sample=result,
            draw_gt=False,
            wait_time=0,
        )

        return visualizer.get_image()