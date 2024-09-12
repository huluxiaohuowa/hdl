import requests

import torch
import numpy as np
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

from ..database_tools.connect import conn_redis


# url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
# image = Image.open(requests.get(url, stream=True).raw)
__all__ = [
    "ImgHandler"
]


class ImgHandler:
    def __init__(
        self,
        model_path,
        redis_host,
        redis_port,
        device: str = None
    ) -> None:
        if device is None:
            self.device = torch.device("cuda") \
                if torch.cuda.is_available() \
                else torch.device("cpu")
        else:
            self.device = device

        self.model = ChineseCLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = ChineseCLIPProcessor.from_pretrained(model_path)
        self.redis_host = redis_host
        self.redis_port = redis_port
        self._redis_conn = None


    @property
    def redis_conn(self):
        if self._redis_conn is None:
            self._redis_conn = conn_redis(self.redis_host, self.redis_port)
        return self._redis_conn

    def get_img_features(self, images, **kwargs):
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            **kwargs
        ).to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / \
            image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def get_text_features(
        self,
        texts,
        **kwargs
    ):
        inputs = self.processor(
            text=texts,
            padding=True,
            return_tensors="pt",
            **kwargs
        ).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = text_features / \
            text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    def get_text_img_sims(
        self,
        texts,
        images,
        **kwargs
    ):
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            **kwargs
        ).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)
        return probs

