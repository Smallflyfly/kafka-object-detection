#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/02/26
"""
import time

import cv2
import imutils
import torch
from kafka import KafkaProducer
from kafka.errors import KafkaError

from cfg.config import cfg_mnet
from models.retinaface import RetinaFace
from utils.utils import image_process, load_model, process_face_data
import numpy as np


kafka_producer = KafkaProducer(bootstrap_servers='42.193.174.78:9092')
topic = 'face-detection'

retina_trained_model = './weights/mobilenet0.25_Final.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cfg = cfg_mnet
retina_net = RetinaFace(cfg=cfg, phase='test')
retina_net = load_model(retina_net, retina_trained_model, False)
retina_net = retina_net.to(device)
retina_net.eval()


def detection(im):
    resize = 1
    im, im_width, im_height, scale = image_process(im, device)
    loc, conf, landms = retina_net(im)
    result_data = process_face_data(cfg, im, im_height, im_width, loc, scale, conf, landms, resize)
    return result_data


def kafka_send(topic, data):
    kafka_producer.send(topic, data)


def producer():

    video = cv2.VideoCapture(0)
    time.sleep(5)

    count = 1

    while video.isOpened():
        success, frame = video.read()
        frame = imutils.resize(frame, width=640, height=640)
        if not success:
            break
        face_result = detection(frame)
        if face_result is None:
            continue
        for det in face_result:
            xmin, ymin, xmax, ymax, conf = det
            # xmin, ymin, xmax, ymax = int(xmin) * 4, int(ymin) * 4, int(xmax) * 4, int(ymax) * 4
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 1)
        # cv2.imshow('im', frame)
        # data = cv2.imencode('.jpg', frame)[1].tobytes()
        data = np.asarray(face_result).astype(np.float).tobytes()
        count += 1
        if count % 10 == 0:
            kafka_send(topic, data)
            count = 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # try:
        #     future.get(timeout=10)
        # except KafkaError as e:
        #     print(e)
        #     break

        print('.', end='', flush=True)

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    producer()
