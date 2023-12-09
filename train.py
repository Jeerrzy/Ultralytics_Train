#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : train.py
# @Time     : 2023/11/30 21:56
# @Project  : ultralytics_train


from ultralytics import YOLO
from multiprocessing import freeze_support


if __name__ == "__main__":
    freeze_support()
    # 加载模型
    model = YOLO('cfg/yolov8n-pose.yaml').load('yolov8n-pose.pt')  # 从YAML构建并传输权重
    # 训练模型
    results = model.train(data='cfg/CPUWormDataset.yaml',
                          epochs=200,
                          imgsz=640,
                          batch=4,
                          workers=0
                          )

