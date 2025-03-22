# Modelo utilizado https://huggingface.co/stabilityai/stable-diffusion-2-1
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = "stabilityai/stable-diffusion-2-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_images(prompts, num_images=1, save_dir='/images'):
    for prompt in prompts:
        for _ in range(num_images):
            print(f"Generating image for prompt: {prompt}")
            image = run_model(prompt)
            print(f"Saving image for prompt: {prompt}")
            image.save(f'{save_dir}/{prompt}.png')


def run_model(prompt):
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    image = pipe(prompt).images[0]
    return image






    

