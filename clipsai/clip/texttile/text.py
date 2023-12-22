"""
Defines an abstract class for embedding text.
"""
# standard library imports
import abc

# 3rd party imports
import torch


class TextEmbedder(abc.ABC):
    """
    An abstract class defining classes that embed text.
    """

    @abc.abstractmethod
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
        pass
