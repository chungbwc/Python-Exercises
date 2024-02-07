
import torch
import PIL
import PIL.Image
import numpy as np
import diffusers
import cv2
import os

from PIL import Image
from diffusers import UNet2DModel
from diffusers import AutoPipelineForText2Image
from diffusers import DDPMPipeline
from diffusers import DDPMScheduler

myDir = os.getcwd()

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# pipeline = AutoPipelineForText2Image.from_pretrained(
#    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
# ).to("cuda")

# pipeline = AutoPipelineForText2Image.from_pretrained(
#    "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
# ).to("cuda")

pipeline.enable_model_cpu_offload()

prompt = "A beautiful photo of Li Tak Wai eating ice cream at home happily with a cat, detailed, 8k"

image = pipeline(prompt=prompt,
                 height=512,
                 width=512,
                 ).images[0]

# image = pipeline(
#    prompt
# ).images[0]

image = np.array(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
print(image.shape)
cv2.imshow("result", image)
fileName = os.path.join(myDir, "data", "poohpooh.png")
cv2.imwrite(fileName, image)
cv2.waitKey(0)
