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


def review_sd15(prompt, negative_prompt,w,h):
    pipe = StableDiffusionPipeline.from_pretrained(
        env.sd15_path, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    image = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                    height=w,
                    width=h,
                    num_inference_steps=50,).images[0]
    return image


def review_sd2(prompt, negative_prompt,w,h):
    repo_id = env.sd2_path
    pipe = DiffusionPipeline.from_pretrained(
        repo_id, torch_dtype=torch.float16, revision="fp16"
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    prompt = prompt
    image = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                    height=w,
                    width=h,
                    num_inference_steps=50,).images[0]
    return image


def review_sdxl(prompt, negative_prompt,w,h):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        env.sdxl_path, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    prompt = prompt
    image = pipe(prompt=prompt,
                 negative_prompt=negative_prompt,
                    height=w,
                    width=h,
                    num_inference_steps=50,
                 ).images[0]
    return image


def review_sc(prompt, negative_prompt,w,h):
    prompt = prompt
    negative_prompt = negative_prompt

    prior = StableCascadePriorPipeline.from_pretrained(
        env.sc_prior_path, variant="bf16", torch_dtype=torch.bfloat16
    )
    decoder = StableCascadeDecoderPipeline.from_pretrained(
        env.sc_decoder_path, variant="bf16", torch_dtype=torch.float16
    )

    prior.enable_model_cpu_offload()
    prior_output = prior(
        prompt=prompt,
        height=w,
        width=h,
        negative_prompt=negative_prompt,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        num_inference_steps=50,
    )

    decoder.enable_model_cpu_offload()
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings.to(torch.float16),
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=10,
    ).images[0]
    return decoder_output


def main():
    width = 1024
    height = 1024
    prompt = "A beautiful sunset"
    negative_prompt = ""
    png_15 = review_sd15(prompt, negative_prompt,width,height)
    png_2 =  review_sd2(prompt, negative_prompt,width,height)
    png_xl = review_sdxl(prompt, negative_prompt,width,height)
    png_sc = review_sc(prompt, negative_prompt,width,height)
    png_15.save("sd15.png")
    png_2.save("sd2.png")
    png_xl.save("sdxl.png")
    png_sc.save("sc.png")

if __name__ == "__main__":
    main()
