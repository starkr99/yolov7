import base64
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadImagesFromBase64
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class Detect :
    opt = {
        'weights': 'yolov7.pt',
        'source': 'inference/images',
        'img_size': 640,
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        'device': '',
        'view_img': False,
        'save_txt': False,
        'save_conf': False,
        'nosave': False,
        'classes': None,
        'agnostic_nms': False,
        'augment': False,
        'update': False,
        'project': 'runs/detect',
        'name': 'exp',
        'exist_ok': False,
        'no_trace': False
    }

    def __init__(self, weights='yolov7.pt', source='inference/images', img_size=640, 
                    conf_thres=0.25, iou_thres=0.45, device='', view_img=False, save_txt=False, 
                    save_conf=False, nosave=False, classes=None, agnostic_nms=False, augment=False, 
                    update=False, project='runs/detect', name='exp', exist_ok=False, no_trace=False):
        
        self.weights = weights
        self.source = source
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.no_trace = no_trace


        self.trace = not self.no_trace
        # Initialize
        set_logging()

        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, device, self.imgsz)

        if self.half:
            self.model.half()  # to FP16

        self.classify = False
        # Second-stage classifier
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]



    def detectFromString(self, source):
        dataset = LoadImagesFromBase64(source, img_size=self.imgsz, stride=self.stride)
        print(self.device)
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()
        ret = []
        for _, img, im0s, _ in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0, frame = '', im0s, 0

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # im0s to base64encoded string and push to ret
            #cv2.imwrite('test.jpg', im0s)
            # convert im0s to jpeg and encode to base64
            ret.append(base64.b64encode(cv2.imencode('.jpg', im0s)[1].tostring()).decode('utf-8')
)

        print(f'Done. ({time.time() - t0:.3f}s)')
        return ret
