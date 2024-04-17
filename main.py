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


def review_sd15():
    pipe = StableDiffusionPipeline.from_pretrained(
        env.sd15_path, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]


def review_sd2():
    repo_id = env.sd2_path
    pipe = DiffusionPipeline.from_pretrained(
        repo_id, torch_dtype=torch.float16, revision="fp16"
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    prompt = "High quality photo of an astronaut riding a horse in space"
    image = pipe(prompt, num_inference_steps=25).images[0]
    image


def review_sdxl():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        env.sdxl_path, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]


def review_sc():
    prompt = "an image of a shiba inu, donning a spacesuit and helmet"
    negative_prompt = ""

    prior = StableCascadePriorPipeline.from_pretrained(
        env.sc_prior_path, variant="bf16", torch_dtype=torch.bfloat16
    )
    decoder = StableCascadeDecoderPipeline.from_pretrained(
        env.sc_decoder_path, variant="bf16", torch_dtype=torch.float16
    )

    prior.enable_model_cpu_offload()
    prior_output = prior(
        prompt=prompt,
        height=1024,
        width=1024,
        negative_prompt=negative_prompt,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        num_inference_steps=20,
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
    decoder_output.save("cascade.png")


def main():
    review_sd15()
    review_sd2()
    review_sdxl()
    review_sc()


if __name__ == "__main__":
    main()
