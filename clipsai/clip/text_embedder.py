"""
Embed text using the Roberta model.
"""
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
        self.__model = SentenceTransformer("all-roberta-large-v1")

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
