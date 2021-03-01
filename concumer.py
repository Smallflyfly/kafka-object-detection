#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/02/26
"""
import argparse
from io import BytesIO

import cv2
from imutils.video import FPS
from kafka import KafkaConsumer
import numpy as np


def show(server, topic):
    consumer = KafkaConsumer(topic, bootstrap_servers=server)
    for message in consumer:
        buf_str = np.array(message.value).tostring()
        nparr = np.asarray(bytearray(buf_str), np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(decoded)
        cv2.imshow('came', decoded)
        # print(decoded)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # fps.update()

    # fps.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='face detection')
    parser.add_argument('--server', default='localhost:9092', type=str, help='kafka server')
    parser.add_argument('--topic', default='face-detection', type=str, help='kafka topic')

    args = parser.parse_args()
    server = args.server
    topic = args.topic
    show(server, topic)