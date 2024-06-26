import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers import DPMSolverMultistepScheduler
import os
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import random
import NDIlib as ndi
import sys
import time

def main():
    if not ndi.initialize():
        return 0


    send_settings = ndi.SendCreate()
    send_settings.ndi_name = 'ndi-python-VideoTest'

    ndi_send = ndi.send_create(send_settings)

    video_frame = ndi.VideoFrameV2()

    #ndiStreamCap = cv2.VideoCapture("./test/VID-20240612-WA0001.mp4")
    ndiStreamCap = cv2.VideoCapture(0)

    start = time.time()
    while time.time() - start < 60 * 30:
        start_send = time.time()
        for _ in reversed(range(200)):
            ret, img = ndiStreamCap.read()

            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

                video_frame.data = img
                video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX

                ndi.send_send_video_v2(ndi_send, video_frame)

        print('200 frames sent, at %1.2ffps' % (200.0 / (time.time() - start_send)))

    ndi.send_destroy(ndi_send)

    ndi.destroy()

    return 0


if __name__ == "__main__":
    sys.exit(main())