import re


class BEEmbedder():
    def __init__(
        self,
        emb_name: str = "bge",
        emb_dir: str = None,
        device: str = 'cuda',
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> None:
        """Initializes the object with the specified embedding name and directory.
        
        Args:
            emb_name (str): The name of the embedding. Defaults to "bge".
            emb_dir (str): The directory path for the embedding model.
        
        Returns:
            None
        """
        self.emb_name = emb_name
        self.emb_dir = emb_dir
        self.batch_size = batch_size
        
        self.model_kwargs = {'device': device}
        self.encode_kwargs = {
            'batch_size': self.batch_size,
            'normalize_embeddings': True,
            'show_progress_bar': False
        }

        if "bge" in emb_name.lower():
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(
                emb_dir,  
                use_fp16=True
            )
        elif "bce" in emb_name.lower():
            from BCEmbedding import EmbeddingModel
            self.model = EmbeddingModel(
                model_name_or_path=emb_dir,
                use_fp16=True
            )
    
    def encode(
        self,
        sentences,
    ):
        """Encode the input sentences using the model.
        
            Args:
                sentences (list): List of sentences to encode.
        
            Returns:
                numpy.ndarray: Encoded representation of the input sentences.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        output = self.model.encode(
            sentences,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=self.batch_size,
            max_length=self.max_length
        )
        if "bge" in self.emb_name.lower():
            return output["dense_vecs"]
        return output
    
    def sim(
        self,
        sentences_1,
        sentences_2
    ):
        """Calculate the similarity between two sets of sentences.
        
            Args:
                sentences_1 (list): List of sentences for the first set.
                sentences_2 (list): List of sentences for the second set.
        
            Returns:
                float: Similarity score between the two sets of sentences.
        """
        output_1 = self.encode(sentences_1)
        output_2 = self.encode(sentences_2)
        similarity = output_1 @ output_2.T
        return similarity


class HFEmbedder():
    def __init__(
        self,
        emb_dir: str = None,
        device: str = 'cuda',
        trust_remote_code: bool = True,
        *args, **kwargs
    ) -> None:
        """Initialize the class with the specified parameters.
        
        Args:
            emb_dir (str): Directory path to the embeddings.
            device (str): Device to be used for computation (default is 'cuda').
            trust_remote_code (bool): Whether to trust remote code (default is True).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                - modules: Optional[Iterable[torch.nn.modules.module.Module]] = None,
                - device: Optional[str] = None,
                - prompts: Optional[Dict[str, str]] = None,
                - default_prompt_name: Optional[str] = None,
                - cache_folder: Optional[str] = None,
                - revision: Optional[str] = None,
                - token: Union[str, bool, NoneType] = None,
                - use_auth_token: Union[str, bool, NoneType] = None,
                - truncate_dim: Optional[int] = None,
        
        Returns:
            None
        """

        from sentence_transformers import SentenceTransformer
        
        self.device = device
        self.emb_dir = emb_dir

        self.model = SentenceTransformer(
            emb_dir,
            device=device,
            trust_remote_code=trust_remote_code,
            *args, **kwargs
        ).half()
        # self.model = model.half()
    
    def encode(
        self,
        sentences: list[str],
        *args, **kwargs
    ):
        """Encode the input sentences using the model.
        
        Args:
            sentences (list[str]): List of input sentences to encode.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                - prompt_name: Optional[str] = None,
                - prompt: Optional[str] = None,
                - batch_size: int = 32,
                - show_progress_bar: bool = None,
                - output_value: Optional[Literal['sentence_embedding', 'token_embeddings']] = 'sentence_embedding',
                - precision: Literal['float32', 'int8', 'uint8', 'binary', 'ubinary'] = 'float32',
                - convert_to_numpy: bool = True,
                - convert_to_tensor: bool = False,
                - device: str = None,
                - normalize_embeddings: bool = False,
        
        Returns:
            output: Encoded representation of the input sentences.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        if kwargs.get("convert_to_tensor", False) is True:
            kwargs["device"] = self.device    
        output = self.model.encode(
            sentences,
            *args, **kwargs
        )
        return output

    def sim(
        self,
        sentences_1,
        sentences_2,
        *args, **kwargs
    ):
        """Calculate the similarity between two sets of sentences.
        
            Args:
                sentences_1 (list): List of sentences for the first set.
                sentences_2 (list): List of sentences for the second set.
                *args: Additional positional arguments to be passed to the encode function.
                **kwargs: Additional keyword arguments to be passed to the encode function.
        
            Returns:
                numpy.ndarray: Similarity matrix between the two sets of sentences.
        """
        output_1 = self.encode(sentences_1, *args, **kwargs)
        output_2 = self.encode(sentences_2, *args, **kwargs)
        similarity = output_1 @ output_2.T
        return similarity


def get_n_tokens(
    paragraph,
    model: str = ""
):
    """Get the number of tokens in a paragraph using a specified model.
    
    Args:
        paragraph (str): The input paragraph to tokenize.
        model (str): The name of the model to use for tokenization. If None, a default CJK tokenization will be used.
    
    Returns:
        int: The number of tokens in the paragraph based on the specified model or default CJK tokenization.
    """
    if model == "":
        cjk_regex = re.compile(u'[\u1100-\uFFFDh]+?')
        trimed_cjk = cjk_regex.sub( ' a ', paragraph, 0)
        return len(trimed_cjk.split())
    else:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(paragraph))
        return num_tokens