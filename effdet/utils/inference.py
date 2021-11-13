import sys
sys.path.append("E:/Users/darkb/OneDrive/Documentos/EIE/Tesis/Pruebas_de_codigos/Yet-Another-EfficientDet-Pytorch")
import torch
from torch.backends import cudnn

from effdet.backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np

from effdet.efficientdet.utils import BBoxTransform, ClipBoxes
from effdet.utils.utils import preprocess, invert_affine, postprocess

def init_effdet_model(weight, obj_list, coef=2, use_cuda=True):


    cudnn.fastest = True
    cudnn.benchmark = True

    model = EfficientDetBackbone(compound_coef=coef, num_classes=len(obj_list),

                                # replace this part with your project's anchor config
                                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    # model.load_state_dict(torch.load('E:/Users/darkb/OneDrive/Documentos/EIE/Tesis/Pruebas_de_codigos/Yet-Another-EfficientDet-Pytorch/weights/efficientdet-d3_198_33400.pth'))
    model.load_state_dict(torch.load(weight))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
  
    return model

def inference_effdet_model(model, img, coef=2, threshold=0.6, use_cuda=True):

    threshold = 0.6
    iou_threshold = 0.1

    force_input_size = None  # set None to use default size
    
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[coef] if force_input_size is None else force_input_size

    framed_imgs, framed_metas = preprocess(img, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    out = invert_affine(framed_metas, out)


    return out[0]
