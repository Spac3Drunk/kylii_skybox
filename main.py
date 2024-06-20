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

from opencvFunc import get_left_half, get_right_half, saveImageList, split_for_ratio_top, vShapeThis


model_id = './revAnimated_v2Rebirth/'
#model_id = "rubbrband/revAnimated_v2Rebirth"

# Function to load pipeline with textual inversions
def load_pipeline(pipeline_class, model_id):
    pipe = pipeline_class.from_pretrained(model_id, torch_dtype=torch.float16)
    embeddings = os.listdir('embeddings')
    for file in embeddings:
        tmpPath = "./embeddings/" + file
        triggerWord = file.split(".")[0]
        print(triggerWord)
        pipe.load_textual_inversion(tmpPath, triggerWord)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

device = "cuda"

def unload_pipeline(pipe):
    # Move the pipeline to CPU
    pipe.to("cpu")
    # Delete the pipeline
    del pipe
    # Clear the CUDA cache
    torch.cuda.empty_cache()

def main():
    # Initialize pipelines
    pipe = load_pipeline(StableDiffusionPipeline, model_id)
    pipeInp = load_pipeline(StableDiffusionInpaintPipeline, model_id)

    inpMask = Image.open("mask.png").convert("RGB")
    generator = torch.manual_seed(random.randint(0, 18446744073709551615))

    prompt = "a mesmerizing digital painting depicting a magical forest within majestic mountains. Whirlwinds should animate the colorful foliage, while an aura of magic infuses the atmosphere with a sense of wonder and enchantment. The scene rendered with high-quality computer graphics, utilizing a vibrant and captivating color scheme to immerse the viewer in this fantastical world., violet, magic whirlwinds"
    neg = "easynegative, By bad artist -neg, bad_prompt_version2-neg, badhandv4, bad-hands-5, (low quality, worst quality,NSFW:1.4), negativeXL_D, text, (signature:1.5), watermark, extra limbs, (interlocked fingers, badly drawn hands and fingers, anatomically incorrect hands), blurry, blurry background"
    h = 960
    w = 960
    steps = 20
    guidance = 7
    num_images = 3

    #groundInpMask = Image.open("groundMask.png").convert("RGB")
    groundPrompt = "360, from above, aerial view from a drone perspective,"
    groundNegPrompt = "sky, mountain, cloud, "

    pipe = pipe.to(device)
    raw_Images = pipe(
        prompt,
        negative_prompt=neg,
        height=h,
        width=w,
        num_inference_step=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        generator=generator,
    ).images

    unload_pipeline(pipe)

    opencvImages = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in raw_Images]

    # Create cropped images
    croppedOpImages = [
        get_left_half(opencvImages[0]),
        cv2.hconcat([get_right_half(opencvImages[0]), get_left_half(opencvImages[1])]),
        cv2.hconcat([get_right_half(opencvImages[1]), get_left_half(opencvImages[2])]),
        get_right_half(opencvImages[2])
    ]

    # Inpaint images
    pipeInp = pipeInp.to(device)
    inpantedImages = [
        croppedOpImages[0],
        pipeInp(
            prompt=prompt,
            negative_prompt=neg,
            height=h,
            width=w,
            image=Image.fromarray(cv2.cvtColor(croppedOpImages[1], cv2.COLOR_BGR2RGB)),
            mask_image=inpMask,
            num_inference_steps=15,
            guidance_scale=guidance,
            generator=generator,
            strength=0.95
        ).images[0],
        pipeInp(
            prompt=prompt,
            negative_prompt=neg,
            height=h,
            width=w,
            image=Image.fromarray(cv2.cvtColor(croppedOpImages[2], cv2.COLOR_BGR2RGB)),
            mask_image=inpMask,
            num_inference_steps=15,
            guidance_scale=guidance,
            generator=generator,
            strength=0.95
        ).images[0],
        croppedOpImages[3]
    ]
    opencvInpantedImages = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) if isinstance(img, Image.Image) else img for img in inpantedImages]
    unload_pipeline(pipeInp)

    # get images back to original size
    almostFinalImages = [
        cv2.hconcat([opencvInpantedImages[0], get_left_half(opencvInpantedImages[1])]),
        cv2.hconcat([get_right_half(opencvInpantedImages[1]), get_left_half(opencvInpantedImages[2])]),
        cv2.hconcat([get_right_half(opencvInpantedImages[2]), opencvInpantedImages[3]])
    ]

    finalTopImages = []
    finalBotImages = []
    for img in almostFinalImages:
        tmp = split_for_ratio_top(img, 235, 250)
        finalTopImages.append(tmp[0])
        finalBotImages.append(tmp[1])

    # Display final images
    #for i, img in enumerate(finalTopImages):
        #cv2.imshow(f'image{i}', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) if isinstance(img, Image.Image) else img)

    #bottom image
    rotated_final_bot_images = []
    rotated_final_bot_images.append(cv2.rotate(finalBotImages[0], cv2.ROTATE_90_COUNTERCLOCKWISE))
    #rotated_final_bot_images.append(vShapeThis(finalBotImages[1]))
    rotated_final_bot_images.append(finalBotImages[1])
    rotated_final_bot_images.append(cv2.rotate(finalBotImages[2], cv2.ROTATE_90_CLOCKWISE))
    """
    # Display final rotated bottom images
    for i, img in enumerate(rotated_final_bot_images):
        cv2.imshow(f'image_bot_rotated_{i}', img)
    """
        # Create a blank canvas
    canvas = np.zeros((w, w, 3), dtype=np.uint8)

        # Paste the rotated bottom images onto the canvas
    canvas[:rotated_final_bot_images[0].shape[0], :rotated_final_bot_images[0].shape[1]] = rotated_final_bot_images[0]
    canvas[:rotated_final_bot_images[2].shape[0], -rotated_final_bot_images[2].shape[1]:] = rotated_final_bot_images[2]
    canvas[:rotated_final_bot_images[1].shape[0], :rotated_final_bot_images[1].shape[1]] = rotated_final_bot_images[1]

        # Display the final bottom image
    #cv2.imshow('Merged Image', canvas)

    # Panoramic image
    fullPanoImg = cv2.hconcat(finalTopImages)
    #cv2.imshow('Concatenated Image', fullPanoImg)

    # Save final images
    finalSavedImages = finalTopImages
    finalSavedImages.append(canvas)
    saveImageList(finalSavedImages)

    if not ndi.initialize():
        return 0

    send_settings = ndi.SendCreate()
    send_settings.ndi_name = 'ndi-python'

    ndi_send = ndi.send_create(send_settings)

    video_frame = ndi.VideoFrameV2()

    ndiStreamImg = cv2.cvtColor(fullPanoImg, cv2.COLOR_BGR2RGBA)
    video_frame.data = ndiStreamImg
    video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBX

    start = time.time()
    while time.time() - start < 60 * 5:
        start_send = time.time()
        for _ in reversed(range(200)):
            ndi.send_send_video_v2(ndi_send, video_frame)

        print('200 frames sent, at %1.2ffps' % (200.0 / (time.time() - start_send)))

    cv2.destroyAllWindows()
    ndi.send_destroy(ndi_send)

    ndi.destroy()

    return 0

if __name__ == "__main__":
    sys.exit(main())