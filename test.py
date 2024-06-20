import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers import DPMSolverMultistepScheduler
import os
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import random

print(torch.__version__)