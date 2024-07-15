class FlagEmbedder():
    def __init__(
        self,
        emb_name: str = "bge",
        emb_dir: str = None
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
        sentences
    ):
        """Encode the input sentences using the model.
        
            Args:
                sentences (list): List of sentences to encode.
        
            Returns:
                numpy.ndarray: Encoded representation of the input sentences.
        """
        output = self.model.encode(
            sentences,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
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
        return similarity.item()
