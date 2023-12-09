#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : predict.py
# @Time     : 2023/11/30 21:48
# @Project  : ultralytics_train


from ultralytics import YOLO


model = YOLO('yolov8n-pose.pt')  # load an official model
# Predict with the model
results = model('./city.jpg', save=True)  # predict on an image


