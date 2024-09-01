import requests
from fastapi import HTTPException
import base64
import os
from PIL import Image
from io import BytesIO

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
STABILITY_ENGINE_ID = "stable-diffusion-v1-6"  # change this by constraint model
STABILITY_API_HOST = 'https://api.stability.ai'


async def process_upscale_image(image_b64: str, prompt: str):
    print("Upscaling image")
    headers = {
        "authorization": f"Bearer {STABILITY_API_KEY}",
        "accept": "image/*"
    }

    # Remove the "data:image/png;base64," prefix if present
    if image_b64.startswith("data:image"):
        image_b64 = image_b64.split(",")[1]

    # Ensure correct padding by adding '=' if necessary
    image_b64 += '=' * (4 - len(image_b64) % 4)

    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))

        # Save the image temporarily to upscale it
        temp_image_path = "./temp_image.png"  # Save in PNG format for consistency
        image.save(temp_image_path)

        print("Image Saved")

        files = {
            "image": open(temp_image_path, "rb"),
        }
        response = requests.post(
            f"{STABILITY_API_HOST}/v2beta/stable-image/upscale/conservative",
            headers=headers,
            files=files,
            data={
                "prompt": prompt,
                "output_format": "webp",
            },
        )

        print("Response", response)

        if response.status_code == 200:
            # Load the upscaled image from the response
            upscaled_image = Image.open(BytesIO(response.content))

            # Convert the upscaled image back to base64
            buffered = BytesIO()
            upscaled_image.save(buffered, format="PNG")
            upscaled_image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return [{"image_b64": upscaled_image_b64}]
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

    except Exception as e:
        print(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

