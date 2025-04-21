#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
EXTRACT_FREQUENCY = 10


def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    frames = []
    while True:
        _, frame = video.read()
        if frame is None:
            break

        if frame_count % EXTRACT_FREQUENCY == 0:
            frames.append(frame)

        frame_count += 1
