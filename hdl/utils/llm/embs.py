# 作者：胡建星（Jianxing Hu）
# 邮箱：j.hu@pku.edu.cn
# 文件：hdl/utils/llm/embs.py
# 说明：大模型调用封装
# 模块功能：文本嵌入（text embedding）模型封装——BEEmbedder 走 FlagEmbedding/BCEmbedding 的 BGE/BCE 模型，
#           HFEmbedder 走 sentence-transformers，两者都提供 encode 编码与 sim 相似度矩阵，另配 get_n_tokens 统计 token 数。
import re


class BEEmbedder():
    """文本嵌入器（text embedding）：按 emb_name 的关键词加载 BGE（FlagEmbedding 的 BGEM3FlagModel）或 BCE（BCEmbedding 的 EmbeddingModel）本地模型，
    两者均以半精度（fp16）推理；sim 用向量点积算相似度，向量需归一化才等价于余弦相似度（cosine similarity）。"""
    def __init__(
        self,
        emb_name: str = "bge",
        emb_dir: str = None,
        device: str = 'cuda',
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> None:
        """Initializes the object with the specified embedding name and directory.
        按 emb_name 中的 "bge"/"bce" 关键字加载对应嵌入模型（权重目录为 emb_dir），编码参数暂存到实例属性。
        注意 max_length 只作为形参存在、未赋给 self.max_length；device 只写进 model_kwargs，未传给模型构造函数。
        
        Args:
            emb_name (str): The name of the embedding. Defaults to "bge".
            emb_dir (str): The directory path for the embedding model.
        
        Returns:
            None
        """
        self.emb_name = emb_name
        self.emb_dir = emb_dir
        self.batch_size = batch_size
        
        # 两组参数字典暂存在实例上：encode_kwargs 里的 normalize_embeddings 使向量为单位长度（点积即余弦相似度），
        # 但下面的 encode 并未读取这两个字典，而是显式传参
        self.model_kwargs = {'device': device}
        self.encode_kwargs = {
            'batch_size': self.batch_size,
            'normalize_embeddings': True,
            'show_progress_bar': False
        }

        # 按名称关键字选择后端库，均用 fp16 推理；两个关键字都不命中时不会赋值 self.model
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
        """把句子编码成嵌入向量：单条字符串会被包成列表，模型同时输出稠密/稀疏两路结果，这里只取稠密向量。
        bge 分支返回 output["dense_vecs"]（numpy 数组，形状 [句子数, 向量维度]）；bce 分支直接返回 model.encode 的原始输出。
        注意 self.max_length 在 __init__ 中未赋值，显式调用会抛 AttributeError。

        Encode the input sentences using the model.
        
            Args:
                sentences (list): List of sentences to encode.
        
            Returns:
                numpy.ndarray: Encoded representation of the input sentences.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        # BGE-M3 的三路输出开关：dense=句向量，sparse=稀疏词权重，colbert=逐 token 多向量；此处只要前两路
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
        """两组文本各自编码后做矩阵乘 output_1 @ output_2.T，得到形状 [len(sentences_1), len(sentences_2)] 的相似度矩阵；
        向量归一化时该点积即余弦相似度。

        Calculate the similarity between two sets of sentences.
        
            Args:
                sentences_1 (list): List of sentences for the first set.
                sentences_2 (list): List of sentences for the second set.
        
            Returns:
                float: Similarity score between the two sets of sentences.
        """
        # 两组分别编码后再转置相乘，逐对比较（不是逐行配对）
        output_1 = self.encode(sentences_1)
        output_2 = self.encode(sentences_2)
        similarity = output_1 @ output_2.T
        return similarity


class HFEmbedder():
    """基于 sentence-transformers 的文本嵌入器（text embedding）：从本地目录或模型名加载 SentenceTransformer 并转成 fp16，
    编码参数（prompt、batch_size、normalize_embeddings 等）全部透传给底层 encode。"""
    def __init__(
        self,
        emb_dir: str = None,
        device: str = 'cuda',
        trust_remote_code: bool = True,
        *args, **kwargs
    ) -> None:
        """加载 SentenceTransformer（权重来自 emb_dir），指定计算设备并按需信任模型仓库自带代码，最后 .half() 转半精度以省显存；
        *args/**kwargs 原样传给 SentenceTransformer 构造函数。

        Initialize the class with the specified parameters.
        
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
        """编码句子为嵌入向量：字符串会被包成单元素列表；convert_to_tensor=True 时把 device 填成构造时的 self.device，
        其余参数（batch_size、normalize_embeddings、precision 等）原样透传，返回值形状与类型由这些参数决定（默认 numpy [句子数, 维度]）。

        Encode the input sentences using the model.
        
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
        # 要求返回 torch.Tensor 时显式指定设备，否则 sentence-transformers 会用默认设备
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
        """两组文本各自编码后算 output_1 @ output_2.T，得到 [len(sentences_1), len(sentences_2)] 的相似度矩阵；
        *args/**kwargs 透传给 encode，需要余弦相似度时须传入 normalize_embeddings=True。

        Calculate the similarity between two sets of sentences.
        
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
    """统计文本 token 数：model 为空串时按中日韩（CJK）逐字近似计数，否则用 tiktoken 按指定模型词表编码后数长度。

    Get the number of tokens in a paragraph using a specified model.
    
    Args:
        paragraph (str): The input paragraph to tokenize.
        model (str): The name of the model to use for tokenization. If None, a default CJK tokenization will be used.
    
    Returns:
        int: The number of tokens in the paragraph based on the specified model or default CJK tokenization.
    """
    if model == "":
        # 字符类 [\u1100-\uFFFD] 覆盖韩文/CJK/全角标点，外加字母 h；量词 +? 惰性匹配，实际每次只匹配 1 个字符
        cjk_regex = re.compile(u'[\u1100-\uFFFDh]+?')
        # count=0 表示全部替换：每个命中的单字符换成独立词 " a "，于是每个 CJK 字符（和每个 h）各计 1 个 token，
        # 其余拉丁词按空白切分各计 1，结果按空白切分取词数
        trimed_cjk = cjk_regex.sub( ' a ', paragraph, 0)
        return len(trimed_cjk.split())
    else:
        # 精确路径：按 OpenAI 模型名取对应编码（encoding），编码后得到 token id 序列，长度为 token 数
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(paragraph))
        return num_tokens