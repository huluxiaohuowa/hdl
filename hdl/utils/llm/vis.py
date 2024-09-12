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
        model_name: str = None,
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
        if model_name is not None:
            self.model_name = model_name
        else:
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
        self.vector_dimension = self.open_clip_cfg["model_cfg"]["embed_dim"]
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
        """Establishes a connection to Redis server if not already connected.

            Returns:
                Redis connection object: A connection to the Redis server.
        """
        if self._redis_conn is None:
            self._redis_conn = conn_redis(self.redis_host, self.redis_port)
        return self._redis_conn

    def get_img_features(
        self,
        images,
        to_numpy = False,
        **kwargs
    ):
        """Get image features using a pretrained model.

        Args:
            images (list): List of image paths.
            to_numpy (bool, optional): Whether to convert the image features to numpy array. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            torch.Tensor or numpy.ndarray: Image features extracted from the model.
        """
        with torch.no_grad(), torch.amp.autocast("cuda"):
            imgs = torch.stack([
                self.preprocess_val(Image.open(image)).to(self.device)
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
        """Get text features from the input texts.

        Args:
            texts (list): List of input texts to extract features from.
            to_numpy (bool, optional): Whether to convert the output features to a numpy array. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the model for encoding text.

        Returns:
            torch.Tensor or numpy.ndarray: Extracted text features.

        Example:
            get_text_features(["text1", "text2"], to_numpy=True)
        """
        with torch.no_grad(), torch.amp.autocast("cuda"):
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
        probs: bool = False,
        to_numpy: bool = False,
        **kwargs
    ):
        """Get the probabilities of text-image associations.

        Args:
            texts (list): List of text inputs.
            images (list): List of image inputs.
            probs (bool, optional): Whether to return probabilities. Defaults to False.
            to_numpy (bool, optional): Whether to convert the output to a numpy array. Defaults to False.
            **kwargs: Additional keyword arguments for feature extraction.

        Returns:
            torch.Tensor or numpy.ndarray: Text-image association probabilities.
        """
        with torch.no_grad(), torch.amp.autocast("cuda"):
            image_features = self.get_img_features(images, **kwargs)
            text_features = self.get_text_features(texts, **kwargs)
            text_probs = (100.0 * image_features @ text_features.T)
            # >3 有关联
            if probs:
                text_probs = text_probs.softmax(dim=-1)
            if to_numpy:
                text_probs = text_probs.cpu().numpy()
        return text_probs

    def get_pics_sims(
        self,
        images1,
        images2,
        to_numpy: bool = False,
        **kwargs
    ):
        """Calculate similarity scores between two sets of images.

            Args:
                images1: First set of images.
                images2: Second set of images.
                to_numpy: Whether to convert the similarity scores to a numpy array (default is False).
                **kwargs: Additional keyword arguments to pass to the get_img_features method.

            Returns:
                torch.Tensor or numpy.ndarray: Similarity scores between the two sets of images.
        """
        with torch.no_grad(), torch.amp.autocast("cuda"):
            img1_feats = self.get_img_features(images1, **kwargs)
            img2_feats = self.get_img_features(images2, **kwargs)
            sims = img1_feats @ img2_feats.T
            if to_numpy:
                sims = sims.cpu().numpy()
            # > 0.9 很相似
            return sims


