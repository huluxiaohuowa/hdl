# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/llm/vis.py
# 说明：大模型调用封装
# 模块功能：多模态输入图像的格式互转与可视化——在 Base64（data URI）、URL、本地文件路径与 PIL Image 之间转换，
#           并按大模型返回的 JSON 框坐标在图上绘制边界框（bounding box）。
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


# Hugging Face Hub 模型名前缀标记（本文件内只定义、未使用）
HF_HUB_PREFIX = "hf-hub:"

def to_img(img_str):
    """
    把图像来源字符串统一转成 PIL Image：按前缀分派——"data:image" 走 Base64 解码，"http" 走网络下载，其余当本地路径打开。
    三种来源都不匹配、或下载响应非 200 时，局部变量 img 不会被赋值。
    Args:
        img_str (str): Base64 data URI、http(s) URL 或本地图片文件路径。
    Returns:
        PIL.Image.Image: 解码后的图像。
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
    把任意图像来源统一成 Base64 data URI 字符串：PIL 对象重新编码，已是 data URI 的原样返回，URL 与本地文件分别走对应转换函数。
    Args:
        img (PIL.Image.Image | str): PIL 图像，或 data URI / URL / 本地路径字符串。
    Returns:
        str: "data:image/<格式>;base64,<...>" 字符串；类型不认识时返回初始空串。
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
    """下载 URL 指向的图片并编码成 data URI：先用 PIL 打开字节流嗅探真实格式，据此拼 MIME 类型，再对原始字节做 Base64。
    非 200 响应抛异常。

    Converts an image from a URL to base64 format.

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
        # 只读字节流判断格式（如 JPEG/PNG），不改变原始数据，Base64 编码的仍是原图字节
        img = Image.open(BytesIO(img_data))
        img_format = img.format.lower()  # Get image format (e.g., jpeg, png)

        # Determine the MIME type based on the format
        mime_type = f"image/{img_format}"

        # Convert the image data to base64
        # 拼成 data URI：头部声明 MIME，正文为原始字节的 Base64（ASCII 解码，可直接嵌进 JSON/HTML）
        img_base64 = f"data:{mime_type};base64," + base64.b64encode(img_data).decode('utf-8')

        return img_base64
    else:
        raise Exception(f"Failed to retrieve image from {image_url}, status code {response.status_code}")


def imgfile_to_base64(img_dir: str):
    """读取本地图片文件的原始字节，用 PIL 嗅探格式得到 MIME，再编码成 data URI（格式与 imgurl_to_base64 一致，只是数据来源是磁盘）。

    Converts an image file to base64 format, supporting multiple formats.

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
    """把 data URI 或纯 Base64 字符串解码成 PIL 图像：split(",")[-1] 去掉 "data:image/...;base64," 头部，解码字节流经 BytesIO 交给 PIL，并统一转成 RGB。

    Converts a base64 encoded image to a PIL image.

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
    """把 PIL 图像编码成 data URI：先无损另存为 PNG 写入内存缓冲区，再对缓冲区字节做 Base64 并加上 "data:image/png;base64," 头部（不论原图格式，输出统一为 PNG）。

    Converts a PIL image to base64 format.

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
    按 JSON 里的边界框（bounding box）坐标在图上画框并输出图像：解析每个目标的类别名与框坐标，换算成像素后画蓝色矩形与红色标签，
    再交给 matplotlib 重绘成 8x8 英寸无边距 PNG 读回为 PIL 图像。
    Args:
        json_data (str | list): JSON 字符串（允许带 ```json 代码块围栏）或已解析的列表，元素形如 {"object": 类别名, "bboxes": [[x1, y1, x2, y2], ...]}，
            坐标按 0~1000 归一化网格给出。
        image: PIL 图像对象，或交给 to_img 解析的来源字符串（data URI / URL / 路径）。注意画框会就地修改传入的 PIL 图像。
        save_path (str | None): 非空时把结果图写到该路径。
    Returns:
        tuple: (带框的 PIL 图像, save_path)；JSON 解析失败时返回 None。

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
    # 大模型常把 JSON 包在 ```json 围栏里，先 strip 再用两次正则剥掉首尾围栏，失败（含截断输出）时打印错误并返回 None
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
            # 坐标按 0~1000 归一化网格给出，乘回真实宽高得到像素坐标（x 用宽、y 用高，故非等比）
            x1 = x1 * width / 1000
            y1 = y1 * height / 1000
            x2 = x2 * width / 1000
            y2 = y2 * height / 1000
            draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=5)
            draw.text((x1, y1), object_type, fill="red", font=font)

    # Plot the image using matplotlib and save it as a PIL Image
    # 用 matplotlib 重绘一遍（8x8 英寸、隐去坐标轴、tight 裁剪无边距），写入内存缓冲区而非磁盘
    buf = BytesIO()
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")  # Hide axes ticks
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    # 指针回到缓冲区开头，否则 Image.open 读到的是文件尾
    buf.seek(0)

    # Load the buffer into a PIL Image and ensure full loading into memory
    pil_image = Image.open(buf)
    pil_image.load()  # Ensure full data is loaded from the buffer

    # Save the image if save_path is provided
    if save_path:
        pil_image.save(save_path)

    buf.close()  # Close the buffer after use

    return pil_image, save_path
