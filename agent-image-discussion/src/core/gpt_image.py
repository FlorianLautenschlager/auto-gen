import os
import requests
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv(override=True)

# You will need to set these environment variables or edit the following values.
endpoint = os.getenv("GPT_IMAGE_1_AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("GPT_IMAGE_1_DEPLOYMENT_NAME")
api_version = os.getenv("GPT_IMAGE_1_OPENAI_API_VERSION")
subscription_key = os.getenv("GPT_IMAGE_1_AZURE_OPENAI_API_KEY")


def decode_and_save_image(b64_data, output_filename):
    image = Image.open(BytesIO(base64.b64decode(b64_data)))
    image.show()
    image.save(output_filename)


def save_all_images_from_response(
    response_data, filename_prefix: str, llmfilename: str
) -> str:

    if not llmfilename.endswith(".png"):
        llmfilename = llmfilename + ".png"

    filenames = []
    for idx, item in enumerate(response_data["data"]):
        b64_img = item["b64_json"]
        filename = f"temp/{filename_prefix}_{idx+1}_{llmfilename}"
        decode_and_save_image(b64_img, filename)
        print(f"[tool] -> Image saved to: '{filename}'")
        filenames.append(filename)
    return filenames[0]


base_path = f"openai/deployments/{deployment}/images"
params = f"?api-version={api_version}"


def generate_image(prompt: str, filename: str) -> str:
    generation_url = f"{endpoint}{base_path}/generations{params}"
    generation_body = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "quality": "medium",
        "output_format": "png",
    }
    generation_response = requests.post(
        generation_url,
        headers={
            "Api-Key": subscription_key,
            "Content-Type": "application/json",
        },
        json=generation_body,
    ).json()
    return save_all_images_from_response(
        generation_response, "generated_image", filename
    )
