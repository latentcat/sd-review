import os
from env import env
import torch
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
)


def review_sd15(pipe, prompt, negative_prompt,w,h):
    image = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                    height=w,
                    width=h,
                    num_inference_steps=50,).images[0]
    return image


def review_sd2(pipe, prompt, negative_prompt,w,h):

    prompt = prompt
    image = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                    height=w,
                    width=h,
                    num_inference_steps=50,).images[0]
    return image


def review_sdxl(pipe, prompt, negative_prompt,w,h):
    
    prompt = prompt
    image = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                    height=w,
                    width=h,
                    num_inference_steps=50,
                 ).images[0]
    return image


def review_sc(prior, decoder, prompt, negative_prompt,w,h):

    prior_output = prior(
        prompt=prompt,
        height=w,
        width=h,
        negative_prompt=negative_prompt,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        num_inference_steps=50,
    )

    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings.to(torch.float16),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=10,
    ).images[0]
    return decoder_output

def run_sd15(prompts, negative_prompt, width, height, out_dir):
    pipe = StableDiffusionPipeline.from_pretrained(
    env.sd15_path, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    i = 0
    for prompt in prompts:
        png_15 = review_sd15(pipe, prompt, negative_prompt,width,height)
        png_15.save(f"{out_dir}sd15_{i}.png")
        i+=1
    del pipe


def run_sd2(prompts, negative_prompt, width, height, out_dir):
    repo_id = env.sd2_path
    pipe = DiffusionPipeline.from_pretrained(
        repo_id, torch_dtype=torch.float16, revision="fp16"
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    i = 0
    for prompt in prompts:
        png_2 =  review_sd2(pipe, prompt, negative_prompt,width,height)
        png_2.save(f"{out_dir}sd2_{i}.png")
        i+=1
    del pipe

def run_sc(prompts, negative_prompt, width, height, out_dir):
    negative_prompt = negative_prompt

    prior = StableCascadePriorPipeline.from_pretrained(
        env.sc_prior_path, variant="bf16", torch_dtype=torch.bfloat16
    )
    prior.enable_model_cpu_offload()

    decoder = StableCascadeDecoderPipeline.from_pretrained(
        env.sc_decoder_path, variant="bf16", torch_dtype=torch.float16
    )
    decoder.enable_model_cpu_offload()

    i = 0
    for prompt in prompts:
        png_xl = review_sc(prior, decoder, prompt, negative_prompt,width,height)
        png_xl.save(f"{out_dir}sc_{i}.png")
        i+=1

    del prior
    del decoder

def run_sdxl(prompts, negative_prompt, width, height, out_dir):
    pipe = StableDiffusionXLPipeline.from_pretrained(
            env.sdxl_path, torch_dtype=torch.float16
        )
    pipe = pipe.to("cuda")
    i = 0

    for prompt in prompts:
        png_xl = review_sdxl(pipe, prompt, negative_prompt,width,height)
        png_xl.save(f"{out_dir}sdxl_{i}.png")
        i+=1

    del pipe


def main():
    with open("prompts.txt", "r") as f:
        prompts = f.readlines()
    width = 1024
    height = 1024
    negative_prompt = ""
    out_dir = "./output/"
    os.makedirs(out_dir, exist_ok=True)

    run_sd15(prompts, negative_prompt, int(width/2), int(height/2), out_dir)
    run_sd2(prompts, negative_prompt, int(width/4*3), int(height/4*3), out_dir)
    run_sdxl(prompts, negative_prompt, width, height, out_dir)
    run_sc(prompts, negative_prompt, width, height, out_dir)


if __name__ == "__main__":
    main()
