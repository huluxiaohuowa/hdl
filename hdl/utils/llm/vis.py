from pathlib import Path
import json

import torch
import numpy as np
from PIL import Image
# from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import open_clip

from ..database_tools.connect import conn_redis


__all__ = [
    "ImgHandler"
]

HF_HUB_PREFIX = "hf-hub:"

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
        ckpt_file = (
            Path(model_path) / Path("open_clip_pytorch_model.bin")
        ).as_posix()

        self.open_clip_cfg = json.load(
            open(Path(model_path) / Path("open_clip_config.json"))
        )
        self.model_name = (
            self.open_clip_cfg['model_cfg']['text_cfg']['hf_tokenizer_name']
            .split('/')[-1]
        )

        self.model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                model_name=self.model_name,
                pretrained=ckpt_file,
                device=self.device,
                # precision=precision
            )
        )
        self.tokenizer = open_clip.get_tokenizer(
            HF_HUB_PREFIX + model_path
        )
        # self.model = ChineseCLIPModel.from_pretrained(model_path).to(self.device)
        # self.processor = ChineseCLIPProcessor.from_pretrained(model_path)
        self.redis_host = redis_host
        self.redis_port = redis_port
        self._redis_conn = None


    @property
    def redis_conn(self):
        if self._redis_conn is None:
            self._redis_conn = conn_redis(self.redis_host, self.redis_port)
        return self._redis_conn

    def get_img_features(
        self,
        images,
        to_numpy = False,
        **kwargs
    ):
        imgs = torch.stack([
            self.preprocess_val(Image.open(image)).unsqueeze(0).to(self.device)
            for image in images
        ])
        img_features = self.model.encode_image(imgs, **kwargs)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        if to_numpy:
            img_features = img_features.cpu().numpy()
        return img_features

    def get_text_features(
        self,
        texts,
        to_numpy = False,
        **kwargs
    ):
        txts = self.tokenizer(
            texts,
            context_length=self.model.context_length
        ).to(self.device)
        txt_features = self.model.encode_text(txts, **kwargs)
        txt_features /= txt_features.norm(dim=-1, keepdim=True)
        if to_numpy:
            txt_features = txt_features.cpu().numpy()
        return txt_features


    def get_text_img_probs(
        self,
        texts,
        images,
        **kwargs
    ):
        image_features = self.get_img_features(images, **kwargs)
        text_features = self.get_text_features(texts, **kwargs)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs


