from FlagEmbedding import BGEM3FlagModel


class FlagEmbedder():
    def __init__(
        self,
        emb_dir: str = None
    ) -> None:
        """Initialize the class with the specified embedding directory.
        
        Args:
            emb_dir (str): The directory containing the embeddings.
            
        Returns:
            None
        """
        self.model = BGEM3FlagModel(
            emb_dir,  
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
        similarity = output_1['dense_vecs'] @ output_2['dense_vecs'].T
        return similarity.item()
