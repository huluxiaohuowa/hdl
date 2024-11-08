from pathlib import Path
import json
import base64
from io import BytesIO
import requests
import uuid
import hashlib

import torch
import numpy as np
from PIL import Image
# from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from transformers import AutoModel
from transformers import AutoTokenizer
import open_clip
# import natsort
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from hdl.jupyfuncs.show.pbar import tqdm
from redis.commands.search.query import Query


# from ..database_tools.connect import conn_redis


HF_HUB_PREFIX = "hf-hub:"


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


class ImgHandler:
    """
    ImgHandler is a class for handling image processing tasks using pretrained models.
    Attributes:
        device_str (str): The device string (e.g., "cpu" or "cuda").
        device (torch.device): The device to run the model on.
        model_path (str): The path to the pretrained model.
        model_name (str): The name of the model.
        model_type (str): The type of the model (e.g., "openclip" or "cpm").
        db_conn: The database connection object.
        num_vec_dim (int): The number of vector dimensions.
        pic_idx_name (str): The name of the picture index.
        open_clip_cfg (dict): The configuration for the OpenCLIP model.
        model: The pretrained model.
        preprocess_train: The preprocessing function for training data.
        preprocess_val: The preprocessing function for validation data.
        tokenizer: The tokenizer for the model.
    Methods:
        __init__(self, model_path, conn=None, model_name: str = None, model_type: str = "openclip", device: str = "cpu", num_vec_dim: int = None, load_model: bool = True) -> None:
            Initializes the ImgHandler class with the specified parameters.
        load_model(self):
            Loads the pretrained model and related configurations.
        get_img_features(self, images, to_numpy=False, **kwargs):
            Gets image features using a pretrained model.
        get_text_features(self, texts, to_numpy=False, **kwargs):
            Gets text features from the input texts.
        get_text_img_probs(self, texts, images, probs=False, to_numpy=False, **kwargs):
            Gets the probabilities of text-image associations.
        get_pics_sims(self, images1, images2, to_numpy=False, **kwargs):
            Calculates similarity scores between two sets of images.
        vec_pics_todb(self, images: list[str], conn=None, print_idx_info=False):
            Saves image features to a Redis database, avoiding duplicates.
        get_pic_idx_info(self, conn=None):
            Gets information about the picture index in the Redis database.
        emb_search(self, emb_query, num_max=3, extra_params=None, conn=None):
            Searches for similar embeddings in the database.
        img_search(self, img, num_max=3, extra_params=None, conn=None):
            Searches for similar images in the database based on the input image.
        """
    def __init__(
        self,
        model_path,
        conn=None,
        model_name: str = None,
        model_type: str = "openclip",
        device: str = "cpu",
        num_vec_dim: int = None,
        load_model: bool = True,
    ) -> None:
        """
        Initializes the visualization utility.
        Args:
            model_path (str): Path to the model.
            conn (optional): Database connection object. Defaults to None.
            model_name (str, optional): Name of the model. Defaults to None.
            model_type (str, optional): Type of the model. Defaults to "openclip".
            device (str, optional): Device to run the model on. Defaults to "cpu".
            num_vec_dim (int, optional): Number of vector dimensions. Defaults to None.
            load_model (bool, optional): Flag to load the model immediately. Defaults to True.
        Returns:
            None
        """
        self.device_str = device
        self.device = torch.device(device)
        self.model_path = model_path
        self.model_name = model_name
        self.model_type = model_type

        self.db_conn = conn
        self.num_vec_dim = num_vec_dim
        self.pic_idx_name = "idx:pic_idx"
        if load_model:
            self.load_model()

    def load_model(self):
        """
        Loads the model and tokenizer based on the specified model type.
        This method supports loading two types of models: "cpm" and "openclip".
        For "cpm":
            - Loads the tokenizer and model using `AutoTokenizer` and `AutoModel` from the Hugging Face library.
            - Sets the model to the specified device.
            - Sets the number of vector dimensions to 2304.
        For "openclip":
            - Loads the model checkpoint and configuration from the specified path.
            - Sets the model name if not already specified.
            - Creates the model and preprocessing transforms using `open_clip.create_model_and_transforms`.
            - Sets the number of vector dimensions based on the configuration if not already specified.
            - Loads the tokenizer using `open_clip.get_tokenizer`.
        Attributes:
            model_type (str): The type of the model to load ("cpm" or "openclip").
            model_path (str): The path to the model files.
            device (str): The device to load the model onto (e.g., "cpu" or "cuda").
            model_name (str, optional): The name of the model (used for "openclip" type).
            num_vec_dim (int, optional): The number of vector dimensions (used for "openclip" type).
            tokenizer: The tokenizer for the model.
            model: The loaded model.
            preprocess_train: The preprocessing transform for training (used for "openclip" type).
            preprocess_val: The preprocessing transform for validation (used for "openclip" type).
            open_clip_cfg (dict): The configuration for the "openclip" model.
        """


        if self.model_type == "cpm":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.num_vec_dim = 2304

        elif self.model_type == "openclip":
            ckpt_file = (
                Path(self.model_path) / Path("open_clip_pytorch_model.bin")
            ).as_posix()
            self.open_clip_cfg = json.load(
                open(Path(self.model_path) / Path("open_clip_config.json"))
            )

            if self.model_name is None:
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
            if self.num_vec_dim is None:
                self.num_vec_dim = self.open_clip_cfg["model_cfg"]["embed_dim"]
            self.tokenizer = open_clip.get_tokenizer(
                HF_HUB_PREFIX + self.model_path
            )

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

        images_fixed = []
        for img in images:
            if isinstance(img, str):
                if img.startswith("data:image"):
                    images_fixed.append(imgbase64_to_pilimg(img))
                elif Path(img).is_file():
                    images_fixed.append(Image.open(img))
            elif isinstance(img, Image.Image):
                images_fixed.append(img)
            else:
                raise TypeError(
                    f"Not supported image type for {type(img)}"
                )

        if self.model_type == "cpm":
            with torch.no_grad():
                img_features = self.model(
                    text=[""] * len(images_fixed),
                    image=images_fixed,
                    tokenizer=self.tokenizer
                ).reps

        if self.model_type == "openclip":
            with torch.no_grad(), torch.amp.autocast(self.device_str):
                imgs = torch.stack([
                    self.preprocess_val(image).to(self.device)
                    for image in images_fixed
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

        if self.model_type == "cpm":
            with torch.no_grad():
                txt_features = self.model(
                    text=texts,
                    image=[None] * len(texts),
                    tokenizer=self.tokenizer
                ).reps
        elif self.model_type == "openclip":
            with torch.no_grad(), torch.amp.autocast(self.device_str):
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
        with torch.no_grad(), torch.amp.autocast(self.device_str):
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
        with torch.no_grad(), torch.amp.autocast(self.device_str):
            img1_feats = self.get_img_features(images1, **kwargs)
            img2_feats = self.get_img_features(images2, **kwargs)
            sims = img1_feats @ img2_feats.T
            if to_numpy:
                sims = sims.cpu().numpy()
            # > 0.9 很相似
            return sims

    def vec_pics_todb(
        self,
        images: list[str],
        conn=None,
        print_idx_info: bool = False,
    ):
        """Save image features to a Redis database, avoiding duplicates.

        Args:
            images (list): A list of image file paths.

        Returns:
            None

        Example:
            vec_pics_todb(images=['image1.jpg', 'image2.jpg'])
        """
        # sorted_imgs = natsort.natsorted(images)
        sorted_imgs = images
        img_feats = self.get_img_features(sorted_imgs, to_numpy=True)
        if conn is None:
            conn = self.db_conn
        pipeline = conn.pipeline()

        for img_file, emb in tqdm(zip(sorted_imgs, img_feats)):
            if img_file.startswith("data:"):
                img_data = img_file
                img_idx = f"pic-{str(uuid.uuid4())}"
            else:
                img_data = imgfile_to_base64(img_file)
                img_idx = f"pic-{img_file}"

            # 使用图片特征生成唯一哈希值
            emb_hash = hashlib.sha256(emb.tobytes()).hexdigest()

            # 检查该哈希值是否已存在，避免重复存储
            if conn.exists(f"pic-hash-{emb_hash}"):
                print(f"Image {img_file} already exists, skipping.")
                continue

            # 存储新图片的特征和数据
            emb = emb.astype(np.float32).tolist()
            emb_json = {
                "emb": emb,
                "data": img_data
            }
            pipeline.json().set(img_idx, "$", emb_json)

            # 将哈希值作为键存储，以便后续检查
            pipeline.set(f"pic-hash-{emb_hash}", img_idx)

        res = pipeline.execute()

        # 定义向量索引的schema
        schema = (
            VectorField(
                "$.emb",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.num_vec_dim,
                    "DISTANCE_METRIC": "COSINE",
                },
                as_name="vector",
            ),
        )
        # 定义索引的配置
        definition = IndexDefinition(
            prefix=["pic-"],
            index_type=IndexType.JSON
        )

        # 检查索引是否已经存在
        try:
            conn.ft(self.pic_idx_name).info()  # 检查索引信息
            print("Index already exists, skipping creation.")
        except Exception:
            # 如果索引不存在，创建新的索引
            res = conn.ft(self.pic_idx_name).create_index(
                fields=schema,
                definition=definition
            )
            print("create_index:", res)

        if print_idx_info:
            print(self.pic_idx_info)

    def get_pic_idx_info(
        self,
        conn=None
    ):
        if conn is None:
            conn = self.db_conn
        res = conn.ping()
        print("redis connected:", res)
        # vector_idx_name = "idx:pic_idx"
        # 从 Redis 数据库中读取索引状态
        info = conn.ft(self.pic_idx_name).info()
        # 获取索引状态中的 num_docs 和 hash_indexing_failures
        num_docs = info["num_docs"]
        indexing_failures = info["hash_indexing_failures"]
        return (
            f"{num_docs} documents indexed with {indexing_failures} failures"
        )

    def emb_search(
        self,
        emb_query,
        num_max: int = 3,
        extra_params: dict = None,
        conn=None
    ):
        """Search for similar embeddings in the database.

        Args:
            emb_query (str): The embedding query to search for.
            num_max (int, optional): The maximum number of results to return. Defaults to 3.
            extra_params (dict, optional): Extra parameters to include in the search query. Defaults to None.

        Returns:
            list: A list of tuples containing the document ID and JSON data for each result.
        """
        query = (
            Query(
                f"(*)=>[KNN {str(num_max)} @vector $query_vector AS vector_score]"
            )
            .sort_by("vector_score")
            .return_fields("$")
            .dialect(2)
        )
        if extra_params is None:
            extra_params = {}
        if conn is None:
            conn = self.db_conn
        result_docs = (
            conn.ft("idx:pic_idx")
            .search(
                query,
                {
                    "query_vector": emb_query
                }
                | extra_params,
            )
            .docs
        )
        results = [
            (result_doc.id, json.loads(result_doc.json))
            for result_doc in result_docs
        ]
        return results

    def img_search(
        self,
        img,
        num_max: int = 3,
        extra_params: dict = None,
        conn=None
    ):
        """Search for similar images in the database based on the input image.

        Args:
            img: Input image to search for similar images.
            num_max: Maximum number of similar images to return (default is 3).
            extra_params: Additional parameters to include in the search query (default is None).

        Returns:
            List of tuples containing the ID and JSON data of similar images found in the database.
        """
        emb_query = self.get_img_features(
            [img], to_numpy=True
        ).astype(np.float32)[0].tobytes()
        if conn is None:
            conn = self.db_conn
        results = self.emb_search(
            emb_query=emb_query,
            num_max=num_max,
            extra_params=extra_params,
            conn=conn
        )
        return results



