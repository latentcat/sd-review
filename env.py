from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


load_dotenv()


class Env(BaseSettings):
    model_config = ConfigDict(extra="ignore")

    sd15_path: str = "runwayml/stable-diffusion-v1-5"
    sd2_path: str = "stabilityai/stable-diffusion-2-base"
    sdxl_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    sc_prior_path: str = "stabilityai/stable-cascade-prior"
    sc_decoder_path: str = "stabilityai/stable-cascade"

    hf_endpoint: str = "https://huggingface.co"
    proxy: str = ""

    stabilityai_api_key: str = ""


env = Env()
