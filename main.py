import io
from PIL import Image, ImageDraw, ImageFont
import os
import requests
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


def review_sd3(prompt, negative_prompt, aspect_ratio):
    response = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers={
            "authorization": f"Bearer {env.stabilityai_api_key}",
            "accept": "image/*",
        },
        files={"none": ""},
        data={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "model": "sd3",
            "negative_prompt": negative_prompt,
            "seed": 0,
            "output_format": "jpeg",
        },
    )

    if response.status_code == 200:
        # 将 bytes 数据转换为 PIL Image 对象
        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes)
        # 保存为临时文件
        return image
    else:
        raise Exception(str(response.json()))


def review_sd15(pipe, prompt, negative_prompt, w, h):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=w,
        width=h,
        num_inference_steps=50,
    ).images[0]
    return image


def review_sd2(pipe, prompt, negative_prompt, w, h):
    prompt = prompt
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=w,
        width=h,
        num_inference_steps=50,
    ).images[0]
    return image


def review_sdxl(pipe, prompt, negative_prompt, w, h):
    prompt = prompt
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=w,
        width=h,
        num_inference_steps=50,
    ).images[0]
    return image


def review_sc(prior, decoder, prompt, negative_prompt, w, h):
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
        png_15 = review_sd15(pipe, prompt, negative_prompt, width, height)
        png_15.save(f"{out_dir}sd15_{i}.png")
        i += 1
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
        png_2 = review_sd2(pipe, prompt, negative_prompt, width, height)
        png_2.save(f"{out_dir}sd2_{i}.png")
        i += 1
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
        png_xl = review_sc(prior, decoder, prompt, negative_prompt, width, height)
        png_xl.save(f"{out_dir}sc_{i}.png")
        i += 1

    del prior
    del decoder


def run_sdxl(prompts, negative_prompt, width, height, out_dir):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        env.sdxl_path, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    i = 0

    for prompt in prompts:
        png_xl = review_sdxl(pipe, prompt, negative_prompt, width, height)
        png_xl.save(f"{out_dir}sdxl_{i}.png")
        i += 1

    del pipe


def run_sd3(prompts, negative_prompt, out_dir):
    i = 0
    for prompt in prompts:
        png_3 = review_sd3(prompt, negative_prompt, "1:1")
        png_3.save(f"{out_dir}sd3_{i}.png")
        i += 1


def merge_images_in_folder(input_folder, font_size=20):
    types = ["sd15", "sd2", "sdxl", "sc", "sd3"]

    # 获取input_folder中的所有文件
    files = [
        f
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
    ]

    # 按后缀数字分组
    groups = {}
    for file in files:
        prefix, num = file.rsplit("_", 1)
        num = num.split(".")[0]
        if num not in groups:
            groups[num] = []
        groups[num].append(file)

    output_folder = os.path.join(input_folder, "merged")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 为了调整字体大小，我们需要指定一个字体文件
    # 如果你没有特定的字体文件，可以使用系统字体，或者使用ImageFont.load_default()
    # font = ImageFont.load_default()
    # 以下是使用truetype加载字体文件的示例，你需要替换成实际的字体文件路径
    font_path = "font.otf"  # 请确保这个路径指向了一个有效的字体文件
    font = ImageFont.truetype(font_path, font_size)

    # 对每组图片进行处理
    for num, group_files in groups.items():
        target_size = (1024, 1024)
        gap_x = 0
        gap_y = 50
        total_width = (target_size[0] + gap_x) * len(group_files) - gap_x
        total_height = target_size[1]
        new_im = Image.new(
            "RGB", (total_width, total_height + gap_y), (255, 255, 255)
        )  # 增加高度以适应更大的字体

        x_offset = 0
        for model_type in types:
            filepath = os.path.join(input_folder, f"{model_type}_{num}.png")
            if not os.path.exists(filepath):
                continue
            im = Image.open(filepath)
            im = im.resize(target_size)
            new_im.paste(im, (x_offset, 50))  # 从50的高度开始，给文件名留更多空间
            x_offset += target_size[0] + gap_x

        # 添加文件名
        draw = ImageDraw.Draw(new_im)
        x_offset = 512
        for model_type in types:
            draw.text(
                (x_offset, 10),
                f"{model_type}",
                fill=(0, 0, 0),
                font=font,
                align="center",
            )
            x_offset += target_size[0] + gap_x

        # 保存新图片
        new_im.save(os.path.join(output_folder, f"merged_{num}.jpg"), quality=90)

    print("图片处理完成。")


def main():
    with open("prompts.txt", "r") as f:
        prompts = f.readlines()
    width = 1024
    height = 1024
    negative_prompt = ""
    out_dir = "./output/"
    os.makedirs(out_dir, exist_ok=True)

    # run_sd3(prompts, negative_prompt, out_dir)
    # run_sd15(prompts, negative_prompt, int(width/2), int(height/2), out_dir)
    # run_sd2(prompts, negative_prompt, int(width/4*3), int(height/4*3), out_dir)
    # run_sdxl(prompts, negative_prompt, width, height, out_dir)
    # run_sc(prompts, negative_prompt, width, height, out_dir)
    merge_images_in_folder(out_dir, font_size=35)


if __name__ == "__main__":
    main()
