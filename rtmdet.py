from mmdet.apis import inference_detector, init_detector
from mmdet.structures import DetDataSample
from mmengine.structures.instance_data import InstanceData
import os
from mmdet.registry import VISUALIZERS
from time import time
import torch

ws_path = os.getcwd()
cur_path = os.path.abspath('.')

class RTMDet:
    def __init__(self):
        config = os.path.join(cur_path, 'weights/rtmdet-ins_s.py')
        checkpoint = os.path.join(cur_path, 'weights/rtmdet-ins_s.pth')
        self.model = init_detector(config, checkpoint, device='cpu')

    def get_prediction(self, image, classes=None, device='cuda:0'):
        pred_start = time()
        pred_result = inference_detector(self.model, image)
        pred_time = time() - pred_start
        
        # extract specified classes
        esc_start = time()
        if (classes is not None):
            # keys = ['kernels', 'labels', 'masks', 'scores', 'bboxes', 'priors']
            kernels = []
            labels = []
            masks = []
            scores = []
            bboxes = []
            priors = []

            if (device == 'cpu'):
                pred_kernels = pred_result.pred_instances.kernels.numpy()
                pred_masks = pred_result.pred_instances.masks.numpy()
                pred_bboxes = pred_result.pred_instances.bboxes.numpy()
                pred_priors = pred_result.pred_instances.priors.numpy()
            else:
                pred_kernels = pred_result.pred_instances.kernels.cpu().numpy()
                pred_masks = pred_result.pred_instances.masks.cpu().numpy()
                pred_bboxes = pred_result.pred_instances.bboxes.cpu().numpy()
                pred_priors = pred_result.pred_instances.priors.cpu().numpy()

            for i in range(len(pred_result.pred_instances['labels'])):
                if (pred_result.pred_instances['labels'][i] in classes):
                    kernels.append(pred_kernels[i])
                    labels.append(pred_result.pred_instances.labels[i].item())
                    masks.append(pred_masks[i])
                    scores.append(pred_result.pred_instances.scores[i].item())
                    bboxes.append(pred_bboxes[i])
                    priors.append(pred_priors[i])

            pred_result.pred_instances = InstanceData(**dict(
                kernels = torch.tensor(kernels),
                labels = torch.tensor(labels),
                masks = torch.tensor(masks),
                scores = torch.tensor(scores),
                bboxes = torch.tensor(bboxes),
                priors = torch.tensor(priors)
            ))
            esc_time = time() - esc_start

        return pred_result, pred_time, esc_time
    
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