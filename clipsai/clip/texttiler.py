"""
Finding topical subsections within text using the TextTiling algorithm.

Notes
-----
- TextTiling was created by Marti A. Hearst in the mid 1990's:
    https://aclanthology.org/J97-1003.pdf
- TextTiling using BERT embeddings was first done June 2021:
    https://arxiv.org/abs/2106.12978
"""
# standard library imports
from collections.abc import Awaitable, Callable
import logging

# current package imports
from .exceptions import TextTilerError

# local package imports
from clipsai.filesys.manager import FileSystemManager
from clipsai.utils.config_manager import ConfigManager
from clipsai.utils.pytorch import (
    max_magnitude_2d,
    get_compute_device,
    assert_compute_device_available,
)
from clipsai.utils.utils import find_missing_dict_keys

# 3rd party imports
import numpy
import torch
import torch.nn.functional as F

BOUNDARY = 1


class TextTiler:
    """
    Tokenize a document into topical sections using the TextTiling algorithm and
    sentence embeddings. This algorithm detects subtopic shifts based on the analysis
    of lexical co-occurrence patterns using sentence-level (or larger) embeddings.

    The process starts by grouping the embeddings into blocks of a fixed size k. Then a
    cosine similarity score is computed and assigned to embedding gaps. The algorithm
    proceeds by detecting the peak differences between these scores and marking them as
    boundaries. The embeddings are then returned within their similar group.
    """

    def __init__(
        self,
        device: str = None,
    ) -> None:
        """
        Parameters
        ----------
        device: str
            PyTorch device to perform computations on. Ex: 'cpu', 'cuda'. Default is
            None (auto detects the correct device)
        """
        if device is None:
            device = get_compute_device()
        assert_compute_device_available(device)
        self._device = device
        self._config_checker = TextTilerConfigManager()
        self._fs_manager = FileSystemManager()

    def text_tile(
        self,
        embeddings: torch.Tensor,
        k: int = 7,
        window_compare_pool_method: str = "mean",
        embedding_aggregation_pool_method: str = "max",
        smoothing_width: int = 3,
        cutoff_policy: str = "high",
    ) -> tuple[list, torch.Tensor]:
        """
        Groups embeddings together using the TextTiling algorithm

        Parameters
        ----------
        embeddings: torch.Tensor
            tensor of (N, E) where N is the number of embeddings and E is
            the dimension of each embedding
        k: int
            the window size for Text Tiling algorithm
        window_compare_pool_method: str
            the method used to pool embeddings within windows (of size k) for comparison
            to adjacent windows.
            Possible values: 'mean', 'max'
        embedding_aggregation_pool_method: str
            the method used to pool embeddings within a segment to create a single
            embedding for the segment.
            Possible values: 'mean', 'max'
        smoothing_width: int
            The width of the window used by the smoothing method
        cutoff_policy: str
            The policy used to determine how dissimilar adjacent embedding windows must
            be to consider them to be from different segments (a boundary).
            Possible values: 'average', 'high', or 'low'

        Returns
        -------
        (list, torch.Tensor)
            - An N length list with 0's and 1's where 1's indicate a boundary between
            embedding i and i+1 in 'embeddings'. The last element in the list is always
            a 1.
            - A tensor of shape (B+1, E) where B is the number of boundaries in
            'boundaries'. Each embedding in the tensor is the pooled embedding of the
            chosen segments determined by the TextTiling algorithm.
        """
        config = {
            "k": k,
            "window_compare_pool_method": window_compare_pool_method,
            "embedding_aggregation_pool_method": embedding_aggregation_pool_method,
            "smoothing_width": smoothing_width,
            "cutoff_policy": cutoff_policy,
        }
        self._config_checker.assert_valid_config(config)

        N, E = embeddings.shape
        # Correct Fixable Inputs
        # k value must be less than the number of embeddings
        if k >= N:
            new_k = max(N // 5, 2)
            logging.warn(
                "{} is not enough embeddings to have gaps for comparison using a k "
                "value of {}. A new  value of {} will be used instead."
                "".format(N, k, new_k)
            )
            k = new_k
        # smoothing width must be less than the number of embeddings
        if smoothing_width >= N:
            smoothing_width = 2  # won't smooth when smoothing_width < 3

        # Textiling Algorithm
        unsmoothed_gap_scores = self._calc_gap_scores(
            embeddings, k, window_compare_pool_method
        )
        gap_scores = self._smooth_scores(unsmoothed_gap_scores, smoothing_width)
        depth_scores = self._calc_depth_scores(gap_scores)
        boundaries = self._identify_boundaries(depth_scores, cutoff_policy)

        # pool embeddings within each group
        pooled_embeddings = self._pool_embedding_groups(
            embeddings, boundaries, embedding_aggregation_pool_method
        )

        return list(boundaries), pooled_embeddings

    def _calc_gap_scores(
        self,
        embeddings: torch.Tensor,
        k: int,
        pool_method: str,
    ) -> torch.Tensor:
        """
        Computes the gap scores between embeddings.

        The gap score is the cosine similarity between the pooled embeddings of the
        left and right windows.

        Parameters
        ----------
        embeddings: torch.Tensor
            contains embeddings of shape (N, E)
            N = number of embeddings
            E = dimension of each embedding
        k: int
            the block size used for Text Tiling Algorithm

        Returns
        -------
        gap_scores: torch.Tensor
            Contains gap scores between each embedding of shape (N-1)
        """
        # define pooling method
        pool = self._get_pool_method(pool_method)

        # compute gap scores
        N, E = embeddings.shape
        num_gaps = N - 1
        gap_scores = torch.empty((num_gaps)).to(self._device)

        for gap in range(num_gaps):
            # define window indices
            left_window_start = max(0, gap - k + 1)
            left_window_end = right_window_start = gap + 1
            right_window_end = min(gap + 1 + k, N)

            # pool left window
            left_window = embeddings[left_window_start:left_window_end]
            pooled_left_window = pool(left_window, dim=0)

            # pool right window
            right_window = embeddings[right_window_start:right_window_end]
            pooled_right_window = pool(right_window, dim=0)

            # compute gap score as cosine similarity between the two windows
            cos_similarity = F.cosine_similarity(
                pooled_left_window, pooled_right_window, dim=0
            )
            gap_scores[gap] = cos_similarity

        return gap_scores

    def _smooth_scores(
        self,
        scores: torch.Tensor,
        smoothing_width: int,
    ) -> torch.Tensor:
        """
        Smooths 'scores' using the smooth function from the SciPy Cookbook

        Parameters
        ----------
        gap_scores: torch.Tensor
            similarity scores computed between each sentence embedding
        smoothing_width: int
            the width of the window used by the smoothing method

        Returns
        -------
        torch.Tensor
            smoothed gap scores
        """
        gap_scores_np_array = scores.cpu().detach().numpy()
        return torch.Tensor(
            list(
                smooth(
                    x=numpy.array(gap_scores_np_array[:]),
                    window_len=smoothing_width,
                    window="flat",
                )
            )
        )

    def _calc_depth_scores(self, gap_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculates the depth of each gap, i.e. the average difference between the left
        and right peaks and the gap's score

        Parameters
        ----------
        gap_scores: torch.Tensor
            similarity scores computed between embeddings

        Returns
        -------
        depth_scores: torch.Tensor
            depth scores computed for each similarity score
        """
        depth_scores = torch.zeros(len(gap_scores)).to(self._device)
        num_gaps = len(gap_scores)

        for gap in range(num_gaps):
            gap_score = gap_scores[gap]
            # find left peak by iterating backward through gap scores
            left_peak = gap_score
            for i in range(gap, -1, -1):
                # move left until the gap score no longer increases
                if gap_scores[i] >= left_peak:
                    left_peak = gap_scores[i]
                else:
                    break

            # find right peak by iterating forward through gap scores
            right_peak = gap_score
            for i in range(gap, len(gap_scores), 1):
                # move right until the gap score no longer increases
                if gap_scores[i] >= right_peak:
                    right_peak = gap_scores[i]
                else:
                    break

            # calculate depth score
            depth_score = (left_peak - gap_score) + (right_peak - gap_score)
            depth_scores[gap] = depth_score

        return depth_scores

    def _identify_boundaries(
        self,
        depth_scores: torch.Tensor,
        cutoff_policy: str,
    ) -> torch.Tensor:
        """
        Identifies boundaries at the peaks of similarity score differences using a
        computed cutoff score

        Parameters
        ----------
        depth_scores: torch.Tensor
            vector of depth scores computed between each word embedding
        cutoff_policy: str
            the policy used to determine the depth score needed to consider a gap a
            boundary

        Returns
        -------
        list
            N length list of 0's and 1's where a 1 at index i indicates a boundary
            after embedding i. The last element in the list is always a 1.
        """
        N = len(depth_scores) + 1
        boundaries = torch.empty((N)).to(self._device)

        avg = torch.mean(depth_scores)
        stdev = torch.std(depth_scores, unbiased=False)

        # set the cutoff policy
        if cutoff_policy == "average":
            cutoff = avg
        elif cutoff_policy == "high":
            cutoff = avg + stdev
        elif cutoff_policy == "low":
            cutoff = avg - stdev
        else:
            err = (
                "cutoff_policy must be 'average', 'high', or 'low' not '{}'"
                "".format(cutoff_policy)
            )
            logging.error(err)
            raise TextTilerError(err)

        # determine boundaries
        for i in range(len(depth_scores)):
            is_boundary = True
            # depth score must exceed cutoff
            if depth_scores[i] <= cutoff:
                is_boundary = False
            # depth score must exceed depth score of both neighbors
            left_neighbor = depth_scores[max(0, i - 1)]
            right_neighbor = depth_scores[min(i + 1, len(depth_scores) - 1)]
            if depth_scores[i] < left_neighbor:
                is_boundary = False
            if depth_scores[i] < right_neighbor:
                is_boundary = False
            if depth_scores[i] == left_neighbor and depth_scores[i] == right_neighbor:
                is_boundary = False

            # mark boundary
            if is_boundary is True:
                boundaries[i] = 1
            else:
                boundaries[i] = 0

        # last embedding is always a boundary
        boundaries[N - 1] = BOUNDARY

        return boundaries

    def _pool_embedding_groups(
        self,
        embeddings: torch.Tensor,
        boundaries: list,
        pool_method: str,
    ) -> torch.Tensor:
        """
        Combines 'embeddings' within the same group, as determined by the boundaries in
        'boundaries, by pooling the embeddings using the 'pool_method'.

        Parameters
        ----------
        embeddings: torch.Tensor
            tensor of shape (N, E) where N is the number of embeddings and E is
            the dimension of each embedding
        boundaries: list
            N length list of 0's and 1's where a 1 at index i indicates a boundary
            after embedding i
        pool_method: str
            the method used to pool embeddings within a group

        Returns
        -------
        torch.Tensor
            the pooled embeddings as a tensor of shape (B+1, E) where B is the number
            of boundaries in 'boundaries'
        """
        # define pooling method
        pool = self._get_pool_method(pool_method)

        # Group embeddings using the given boundaries
        N, E = embeddings.shape
        pooled_embeddings = []
        cur_group = []

        for i in range(0, N):
            # add embedding to the current group of embeddings
            cur_group.append(embeddings[i, :].reshape(1, E))

            # is a boundary
            if boundaries[i] == BOUNDARY:
                # concatenate the list of embeddings in the current group into a tensor
                # of shape (G, E) where G = number of embeddings in the current group
                cur_group = torch.concat(tuple(cur_group))
                # pool the current group of embeddings into a single embedding (1, E)
                # and add the group to the pooled embedding
                pooled_embeddings.append(pool(cur_group, dim=0))
                # reset the current group
                cur_group = []

        pooled_embeddings = torch.stack(tuple(pooled_embeddings))
        return pooled_embeddings

    def _get_pool_method(
        self, pool_method: str
    ) -> Callable[[torch.Tensor], Awaitable[torch.Tensor]]:
        """
        Returns the pooling method given the name of the method

        Parameters
        ----------
        pool_method: str
            the name of the pooling method

        Returns
        -------
        Callable[[torch.Tensor], Awaitable[torch.Tensor]]
            the pooling method
        """
        if pool_method == "mean":
            return torch.mean
        elif pool_method == "max":
            return max_magnitude_2d
        else:
            err = "pool_method must be 'mean' or 'max' not '{}'".format(pool_method)
            logging.error(err)
            raise TextTilerError(err)


# Pasted from the SciPy cookbook: https://www.scipy.org/Cookbook/SignalSmooth
def smooth(x, window_len=3, window="flat"):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    :param x: the input signal
    :param window_len: the dimension of the smoothing window; should be an odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett',
    'blackman'
        flat window will produce a moving average smoothing.

    :return: the smoothed signal

    example::

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

    :see also: numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve,
        scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a
    string
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = numpy.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]

    if window == "flat":  # moving average
        w = numpy.ones(window_len, "d")
    else:
        w = eval("numpy." + window + "(window_len)")

    y = numpy.convolve(w / w.sum(), s, mode="same")

    return y[window_len - 1 : -window_len + 1]


class TextTilerConfigManager(ConfigManager):
    """
    A class for getting information about and validating TextTiler configuration
    settings.
    """

    def __init__(self) -> None:
        """
        Parameters
        ----------
        None
        """
        super().__init__()

    def check_valid_config(
        self,
        texttile_config: dict,
    ) -> str or None:
        """
        Checks that 'texttile_config' contains valid configuration settings.
        Returns None if valid, a descriptive error message if invalid.

        Parameters
        ----------
        texttile_config: dict
            A dictionary containing the configuration settings for TextTiler.

        Returns
        -------
        str or None
            None if the inputs are valid, otherwise an error message.
        """
        # existence check
        required_keys = [
            "cutoff_policy",
            "embedding_aggregation_pool_method",
            "k",
            "smoothing_width",
            "window_compare_pool_method",
        ]
        missing_keys = find_missing_dict_keys(texttile_config, required_keys)
        if len(missing_keys) != 0:
            return "TextTiler missing configuration settings: {}".format(missing_keys)

        setting_checkers = {
            "cutoff_policy": self.check_valid_cutoff_policy,
            "embedding_aggregation_pool_method": self.check_valid_embedding_aggregation_pool_method,
            "k": self.check_valid_k,
            "smoothing_width": self.check_valid_smoothing_width,
            "window_compare_pool_method": self.check_valid_window_compare_pool_method,
        }
        for setting, checker in setting_checkers.items():
            err = checker(texttile_config[setting])
            if err is not None:
                return err

        return None

    def check_valid_k(self, k: int) -> str or None:
        """
        Checks the window size is valid. Returns None if the window size is valid, a
        descriptive error message if invalid.

        Parameters
        ----------
        k: int
            The window size used by TextTiling algorithm

        Returns
        -------
        str or None
            None if the window size is valid, otherwise an error message.
        """
        err = self._type_checker.check_type(k, "k", int)
        if err is not None:
            return err

        if k < 2:
            return "k value must be 2 or greater, not '{}'".format(k)

        return None

    def check_valid_pool_method(self, pool_method: str) -> str or None:
        """
        Checks the pool method is valid. Returns None if the pool method is valid, a
        descriptive error message if invalid.

        Parameters
        ----------
        pool_method: str
            the method used to pool embeddings within windows (of size k) to compare
            windows to each other.
            Possible values: 'mean', 'max'

        Returns
        -------
        str or None
            None if the pool method is valid, otherwise an error message.
        """
        pool_methods = ["mean", "max"]
        if pool_method not in pool_methods:
            return "pool_method must be one of {} not '{}'" "".format(
                pool_methods, pool_method
            )

        return None

    def check_valid_window_compare_pool_method(self, pool_method: str) -> str or None:
        """
        Checks the pool method used to compare adjacent windows is valid. Returns None
        if the pool method is valid, a descriptive error message if invalid.

        Parameters
        ----------
        pool_method: str
            the method used to pool embeddings within windows (of size k) for comparison
            to adjacent windows.
            Possible values: 'mean', 'max'

        Returns
        -------
        str or None
            None if the pool method is valid, otherwise an error message.
        """
        return self.check_valid_pool_method(pool_method)

    def check_valid_embedding_aggregation_pool_method(
        self, pool_method: str
    ) -> str or None:
        """
        Checks the pool method used to aggregate embeddings within a segment is valid.

        Parameters
        ----------
        pool_method: str
            the method used to pool embeddings within a segment to create a single
            embedding for the segment.
            Possible values: 'mean', 'max'

        Returns
        -------
        str or None
            None if the pool method is valid, otherwise an error message.
        """
        return self.check_valid_pool_method(pool_method)

    def check_valid_smoothing_width(self, smoothing_width: int) -> str or None:
        """
        Checks the smoothing width is valid. Returns None if the smoothing width is
        valid, a descriptive error message if invalid.

        Parameters
        ----------
        smoothing_width: int
            The width of the window used by the smoothing method

        Returns
        -------
        str or None
            None if the smoothing width is valid, otherwise an error message.
        """
        err = self._type_checker.check_type(smoothing_width, "smoothing_width", int)
        if err is not None:
            return err

        if smoothing_width < 3:
            return "smoothing_width must be greater than 2, not '{}'" "".format(
                smoothing_width
            )

        return None

    def check_valid_cutoff_policy(self, cutoff_policy: str) -> str or None:
        """
        Checks the cutoff policy is valid. Returns None if the cutoff policy is valid,
        a descriptive error message if invalid.

        Parameters
        ----------
        cutoff_policy: str
            The policy used to determine how dissimilar adjacent embedding windows must
            be to consider them to be from different segments (a boundary).
            Possible values: 'average', 'high', or 'low'

        Returns
        -------
        str or None
            None if the cutoff policy is valid, otherwise an error message.
        """
        cutoff_policies = ["average", "low", "high"]
        if cutoff_policy not in cutoff_policies:
            return "cutoff_policy must be one of {} not '{}'" "".format(
                cutoff_policies, cutoff_policy
            )

        return None

    def check_valid_clip_times(
        self, min_clip_duration: float, max_clip_duration: float
    ) -> str or None:
        """
        Checks the clip times are valid. Returns None if the clip times are valid, a
        descriptive error message if invalid.

        Parameters
        ----------
        min_clip_duration: float
            The minimum clip time in seconds
        max_clip_duration: float
            The maximum clip time in seconds

        Returns
        -------
        str or None
            None if the clip times are valid, otherwise an error message.
        """
        # type check
        self._type_checker.check_type(
            min_clip_duration, "min_clip_duration", (float, int)
        )
        self._type_checker.check_type(
            max_clip_duration, "max_clip_duration", (float, int)
        )

        # minimum clip time
        if min_clip_duration < 0:
            error = "min_clip_duration must be 0 or greater, not {}" "".format(
                min_clip_duration
            )
            return error

        # maximum clip time
        if max_clip_duration <= min_clip_duration:
            error = (
                "max_clip_duration of {} must be greater than "
                "min_clip_duration of {}"
                "".format(max_clip_duration, min_clip_duration)
            )
            return error

        return None
