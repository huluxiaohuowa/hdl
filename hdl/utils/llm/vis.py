import requests

import torch
import numpy as np
from PIL import Image
import redis
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

from ..database_tools.connect import conn_redis


# url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
# image = Image.open(requests.get(url, stream=True).raw)


class ImgHandler:
    def __init__(
        self,
        model_path,
        redis_host,
        redis_port,
    ) -> None:
        self.model = ChineseCLIPModel.from_pretrained(model_path)
        self.processor = ChineseCLIPProcessor.from_pretrained(model_path)
        self.redis_host = redis_host
        self.redis_port = redis_port

    @property
    def redis_conn(self):
        return conn_redis(self.redis_host, self.redis_port)

    def get_img_features(self, images, **kwargs):
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            **kwargs
        )
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
        )
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
        )
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)
        return probs

