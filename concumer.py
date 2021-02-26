#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/02/26
"""
import argparse

import cv2
from imutils.video import FPS
from kafka import KafkaConsumer
import numpy as np


# server = '42.193.174.78:9092'
# topic = 'face-detection'


def show(server, topic):
    consumer = KafkaConsumer(topic, booststrap_servers=server)
    fps = FPS().start()
    for message in consumer:
        decoded = np.frombuffer(message.value, np.int8)
        # decoded = decoded.reshape
        cv2.imshow('came', decoded)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()

    fps.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='face detection')
    parser.add_argument('--server', default='42.193.174.78:9092', type=str, help='kafka server')
    parser.add_argument('--topict', default='face-detection', type=str, help='kafka topic')

    args = parser.parse_args()
    server = args.server
    topic = args.topic
    show(server, topic)