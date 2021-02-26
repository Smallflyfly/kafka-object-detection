#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/02/26
"""
import pickle

import cv2
import numpy as np
import torch

from models.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms


def image_process(im, device):
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    # covert BGR to RGB
    im = im[:, :, ::-1]
    im = np.array(im).astype(int)
    im_width, im_height = im.shape[1], im.shape[0]
    scale = [im_width, im_height, im_width, im_height]
    scale = torch.from_numpy(np.array(scale))
    scale = scale.float()
    scale = scale.to(device)
    im -= (104, 117, 123)
    im = im.transpose((2, 0, 1))
    im = torch.from_numpy(im).unsqueeze(0)
    im = im.float()
    im = im.to(device)
    return im, im_width, im_height, scale


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def process_face_data(cfg, im, im_height, im_width, loc, scale, conf, landms,
                      resize, top_k=5000, nms_threshold=0.4, keep_top_k=750):
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.cuda()
    priors_data = priors.data
    boxes = decode(loc.data.squeeze(0), priors_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).cpu().detach().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), priors_data, cfg['variance'])
    scale_landm = torch.from_numpy(np.array([
        im.shape[3], im.shape[2], im.shape[3], im.shape[2],
        im.shape[3], im.shape[2], im.shape[3], im.shape[2],
        im.shape[3], im.shape[2]
    ]))
    scale_landm = scale_landm.float()
    scale_landm = scale_landm.cuda()
    landms = landms * scale_landm / resize
    landms = landms.cpu().numpy()

    # ignore low score
    inds = np.where(scores > 0.9)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = np.argsort(-scores)[:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do nms
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(float, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K fater NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    dets = np.concatenate((dets, landms), axis=1)

    result_data = dets[:, :5].tolist()

    return result_data