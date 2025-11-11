"""
Embed text using the Roberta model.
"""
# local imports
from clipsai.utils.model_cache import ModelCache

# 3rd party imports
import torch
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    A class for embedding text using the Roberta model.
    """

    def __init__(self) -> None:
        """
        Parameters
        ----------
        None
        """
        # Use ModelCache to eliminate 1-2 minute loading bottleneck
        cache = ModelCache.get_instance()
        self.__model = cache.get_sentence_transformer("all-roberta-large-v1")

    def embed_sentences(self, sentences: list) -> torch.Tensor:
        """
        Creates embeddings for each sentence in sentences

        Parameters
        ----------
        sentences: list
            a list of N sentences

        Returns
        -------
        - sentence_embeddings: torch.tensor
            a tensor of N x E where n is a sentence and e
            is an embedding for that sentence
        """
        return torch.tensor(self.__model.encode(sentences))
