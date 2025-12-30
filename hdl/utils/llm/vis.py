from pathlib import Path
import json
import base64
from io import BytesIO
import requests
# import uuid
# import hashlib

# import torch
# import numpy as np
# from transformers import ChineseCLIPProcessor, ChineseCLIPModel
# from transformers import AutoModel
# from transformers import AutoTokenizer
# import open_clip

from PIL import Image, ImageDraw, ImageFont
import json
import re
import matplotlib.pyplot as plt
# import natsort
from hdl.jupyfuncs.show.pbar import tqdm


# from decord import VideoReader, cpu

import base64
from io import BytesIO
from PIL import Image
import requests
# from ..database_tools.connect import conn_redis


HF_HUB_PREFIX = "hf-hub:"

def to_img(img_str):
    """
    Convert an image source string to a PIL Image object.
    The function supports three types of image sources:
    1. Base64 encoded image strings starting with "data:image".
    2. URLs starting with "http".
    3. Local file paths.
    Args:
        img_str (str): The image source string. It can be a base64 encoded string, a URL, or a local file path.
    Returns:
        PIL.Image.Image: The converted image as a PIL Image object.
    Raises:
        ValueError: If the image source string is not valid or the image cannot be loaded.
    """
    if img_str.startswith("data:image"):
        img = imgbase64_to_pilimg(img_str)
    elif img_str.startswith("http"):
        response = requests.get(img_str)
        if response.status_code == 200:
            # Read the image content from the response
            img_data = response.content

            # Load the image using PIL to determine its format
            img = Image.open(BytesIO(img_data))
    elif Path(img_str).is_file():
        img = Image.open(img_str)
    return img


def to_base64(img):
    """
    Convert an image to a base64 encoded string.

    Args:
        img (Union[Image.Image, str]): The image to convert, which can be a PIL Image object, a base64 string, a URL, or a local file path.

    Returns:
        str: The image encoded as a base64 string.
    """
    img_base64=""

    if isinstance(img, Image.Image):
        img_base64 = pilimg_to_base64(img)
    elif isinstance(img, str):
        if img.startswith("data:image"):
            img_base64 = img
        elif img.startswith("http"):
            img_base64 = imgurl_to_base64(img)
        elif Path(img).is_file():
            img_base64 = imgfile_to_base64(img)
    return img_base64


def imgurl_to_base64(image_url: str):
    """Converts an image from a URL to base64 format.

    Args:
        image_url (str): The URL of the image.

    Returns:
        str: The image file converted to base64 format with appropriate MIME type.
    """
    # Send a GET request to fetch the image from the URL
    response = requests.get(image_url)

    # Ensure the request was successful
    if response.status_code == 200:
        # Read the image content from the response
        img_data = response.content

        # Load the image using PIL to determine its format
        img = Image.open(BytesIO(img_data))
        img_format = img.format.lower()  # Get image format (e.g., jpeg, png)

        # Determine the MIME type based on the format
        mime_type = f"image/{img_format}"

        # Convert the image data to base64
        img_base64 = f"data:{mime_type};base64," + base64.b64encode(img_data).decode('utf-8')

        return img_base64
    else:
        raise Exception(f"Failed to retrieve image from {image_url}, status code {response.status_code}")


def imgfile_to_base64(img_dir: str):
    """Converts an image file to base64 format, supporting multiple formats.

    Args:
        img_dir (str): The directory path of the image file.

    Returns:
        str: The image file converted to base64 format with appropriate MIME type.
    """
    # Open the image file
    with open(img_dir, 'rb') as file:
        # Read the image data
        img_data = file.read()

        # Get the image format (e.g., JPEG, PNG, etc.)
        img_format = Image.open(BytesIO(img_data)).format.lower()

        # Determine the MIME type based on the format
        mime_type = f"image/{img_format}"

        # Convert the image data to base64
        img_base64 = f"data:{mime_type};base64," + base64.b64encode(img_data).decode('utf-8')

    return img_base64


def imgbase64_to_pilimg(img_base64: str):
    """Converts a base64 encoded image to a PIL image.

    Args:
        img_base64 (str): Base64 encoded image string.

    Returns:
        PIL.Image: A PIL image object.
    """
    # Decode the base64 string and convert it back to an image
    img_pil = Image.open(
        BytesIO(
            base64.b64decode(img_base64.split(",")[-1])
        )
    ).convert('RGB')
    return img_pil


def pilimg_to_base64(pilimg):
    """Converts a PIL image to base64 format.

    Args:
        pilimg (PIL.Image): The PIL image to be converted.

    Returns:
        str: Base64 encoded image string.
    """
    buffered = BytesIO()
    pilimg.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_format = 'png'
    mime_type = f"image/{img_format}"
    img_base64 = f"data:{mime_type};base64,{image_base64}"
    return img_base64



def draw_and_plot_boxes_from_json(
    json_data,
    image,
    save_path=None
):
    """
    Parses the JSON data to extract bounding box coordinates,
    scales them according to the image size, draws the boxes on the image,
    and returns the image as a PIL object.

    Args:
        json_data (str or list): The JSON data as a string or already parsed list.
        image_path (str): The path to the image file on which boxes are to be drawn.
        save_path (str or None): The path to save the resulting image. If None, the image won't be saved.

    Returns:
        PIL.Image.Image: The processed image with boxes drawn on it.
    """
    # If json_data is a string, parse it into a Python object
    if isinstance(json_data, str):
        json_data = json_data.strip()
        json_data = re.sub(r"^```json\s*", "", json_data)
        json_data = re.sub(r"```$", "", json_data)
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON data:", e)
            return None
    else:
        data = json_data

    # Open the image
    # try:
    #     img = Image.open(image_path)
    # except FileNotFoundError:
    #     print(f"Image file not found at {image_path}. Please check the path.")
    #     return None
    if not isinstance(image, Image.Image):
        image = to_img(image)
    img = image

    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Use a commonly available font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=25)
    except IOError:
        print("Default font not found. Using a basic PIL font.")
        font = ImageFont.load_default()

    # Process and draw boxes
    for item in data:
        object_type = item.get("object", "unknown")
        for bbox in item.get("bboxes", []):
            x1, y1, x2, y2 = bbox
            x1 = x1 * width / 1000
            y1 = y1 * height / 1000
            x2 = x2 * width / 1000
            y2 = y2 * height / 1000
            draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=5)
            draw.text((x1, y1), object_type, fill="red", font=font)

    # Plot the image using matplotlib and save it as a PIL Image
    buf = BytesIO()
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")  # Hide axes ticks
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Load the buffer into a PIL Image and ensure full loading into memory
    pil_image = Image.open(buf)
    pil_image.load()  # Ensure full data is loaded from the buffer

    # Save the image if save_path is provided
    if save_path:
        pil_image.save(save_path)

    buf.close()  # Close the buffer after use

    return pil_image, save_path
