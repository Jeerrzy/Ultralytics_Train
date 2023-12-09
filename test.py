#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : test.py
# @Time     : 2023/11/30 23:45
# @Project  : ultralytics_train


import cv2
import json
import numpy as np


def get_txt():
    with open('./Raw/annotations/001.json', 'r') as f:
        rawData = json.load(f)
    W, H = 4632, 3488
    bboxes, keypoints = rawData['bboxes'], rawData['keypoints']
    assert len(bboxes) == len(keypoints), print('length match error!')
    l = len(bboxes)
    txt_result = []
    for i in range(l):
        x1, y1, x2, y2 = bboxes[i]
        [[xe1, ye1, _], [xc, yc, _], [xe2, ye2, _]] = keypoints[i]
        txt_result.append(
            [0, (x1+x2) / 2 / W, (y1+y2)/2 / H, (x2 - x1) / W, (y2 - y1) / H, xe1 / W, ye1 / H, xc / W, yc / H, xe2 / W, ye2 / H])
    np.savetxt('./Raw/test.txt', np.array(txt_result), delimiter=' ')


def visualize2yolo(imagePath, txtPath):
    imageData = cv2.imread(imagePath)
    print(imageData.shape)
    imageHeight, imageWidth = imageData.shape[0:2]
    txtData = np.loadtxt(txtPath)
    for (cls, cx, cy, w, h, ep1x, ep1y, cpx, cpy, ep2x, ep2y) in txtData:
        cx *= imageWidth
        w *= imageWidth
        ep1x *= imageWidth
        cpx *= imageWidth
        ep2x *= imageWidth
        cy *= imageHeight
        h *= imageHeight
        ep1y *= imageHeight
        cpy *= imageHeight
        ep2y *= imageHeight
        cv2.rectangle(imageData, (int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2)), (0, 0, 255), 5)
    cv2.imshow('demo', cv2.resize(imageData, (int(imageWidth*0.25), int(imageHeight*0.25))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for image


if __name__ == "__main__":
    visualize2yolo(
        imagePath='./CPUWormDataset/images/train/080.jpg',
        txtPath='./CPUWormDataset/labels/train/080.txt'
    )
