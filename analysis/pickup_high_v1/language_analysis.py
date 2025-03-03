# The code is from https://github.com/facebookresearch/EGG/blob/main/egg/core/language_analysis.py
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections import defaultdict
from typing import Callable, Union

import editdistance
import numpy as np
import torch
from scipy.spatial import distance
from scipy.stats import spearmanr



def gap_mi_first_second(attributes, representations):
    gaps = torch.zeros(representations.size(1))
    non_constant_positions = 0.0

    for j in range(representations.size(1)):
        symbol_mi = []
        h_j = None
        for i in range(attributes.size(1)):
            x, y = attributes[:, i], representations[:, j]
            info = mutual_info(x, y)
            symbol_mi.append(info)

            if h_j is None:
                h_j = calc_entropy(y)

        symbol_mi.sort(reverse=True)
        # print(symbol_mi)
        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()


def entropy_dict(freq_table):
    """
    >>> d = {'a': 1, 'b': 1}
    >>> np.allclose(entropy_dict(d), 1.0)
    True
    >>> d = {'a': 1, 'b': 0}
    >>> np.allclose(entropy_dict(d), 0.0, rtol=1e-5, atol=1e-5)
    True
    """
    t = torch.tensor([v for v in freq_table.values()]).float()
    if (t < 0.0).any():
        raise RuntimeError("Encountered negative probabilities")

    t /= t.sum()
    return -(torch.where(t > 0, t.log(), t) * t).sum().item() / np.log(2)


def calc_entropy(messages):
    """
    >>> messages = torch.tensor([[1, 2], [3, 4]])
    >>> np.allclose(calc_entropy(messages), 1.0)
    True
    """
    freq_table = defaultdict(float)

    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def _hashable_tensor(t):
    if torch.is_tensor(t) and t.numel() > 1:
        t = tuple(t.tolist())
    elif torch.is_tensor(t) and t.numel() == 1:
        t = t.item()
    return t


def mutual_info(xs, ys):
    """
    I[x, y] = E[x] + E[y] - E[x,y]
    """
    e_x = calc_entropy(xs)
    e_y = calc_entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    e_xy = calc_entropy(xys)

    return e_x + e_y - e_xy

class TopographicSimilarity():
    """
    >>> words = ['cat', 'dog', 'pen', 'ben', 'ten']
    >>> mean = [[ord(c) for c in w] for w in words]
    >>> mess = [[ord(c) for c in w] for w in words]
    >>> round(TopographicSimilarity.compute_topsim(mean, mess, 'hamming', 'hamming'), 6)
    1.0
    >>> round(TopographicSimilarity.compute_topsim(
    ...     mean,
    ...     mess,
    ...     'hamming',
    ...     lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2)),  6
    ... )
    1.0
    """

    def __init__(
        self,
        sender_input_distance_fn: Union[str, Callable] = "hamming",
        message_distance_fn: Union[str, Callable] = "edit",
        compute_topsim_train_set: bool = False,
        compute_topsim_test_set: bool = True,
        is_gumbel: bool = False,
    ):

        self.sender_input_distance_fn = sender_input_distance_fn
        self.message_distance_fn = message_distance_fn

        self.compute_topsim_train_set = compute_topsim_train_set
        self.compute_topsim_test_set = compute_topsim_test_set
        assert compute_topsim_train_set or compute_topsim_test_set

        self.is_gumbel = is_gumbel

    @staticmethod
    def compute_topsim(
        meanings: torch.Tensor,
        messages: torch.Tensor,
        meaning_distance_fn: Union[str, Callable] = "hamming",
        message_distance_fn: Union[str, Callable] = "edit",
    ) -> float:

        distances = {
            "edit": lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2),
            "cosine": distance.cosine,
            "hamming": distance.hamming,
            "jaccard": distance.jaccard,
            "euclidean": distance.euclidean,
        }

        meaning_distance_fn = (
            distances.get(meaning_distance_fn, None)
            if isinstance(meaning_distance_fn, str)
            else meaning_distance_fn
        )
        message_distance_fn = (
            distances.get(message_distance_fn, None)
            if isinstance(message_distance_fn, str)
            else message_distance_fn
        )

        assert (
            meaning_distance_fn and message_distance_fn
        ), f"Cannot recognize {meaning_distance_fn} \
            or {message_distance_fn} distances"

        meaning_dist = distance.pdist(meanings, meaning_distance_fn)
        message_dist = distance.pdist(messages, message_distance_fn)

        topsim = spearmanr(meaning_dist, message_dist, nan_policy="raise").correlation

        return topsim



class Disent():
    """
    Callback to compute positional and bago of symbols disentanglement metrics.

    Metrics introduced in "Compositionality and Generalization in Emergent Languages", Chaabouni et al., ACL 2020.

    Two-symbol messages representing two-attribute world. One symbol encodes one attribute:
    in this case, the metric should be maximized:
    >>> samples = 1_000
    >>> _ = torch.manual_seed(0)
    >>> attribute1 = torch.randint(low=0, high=10, size=(samples, 1))
    >>> attribute2 = torch.randint(low=0, high=10, size=(samples, 1))
    >>> attributes = torch.cat([attribute1, attribute2], dim=1)
    >>> messages = attributes
    >>> round(Disent.posdis(attributes, messages), 6)
    0.978656
    >>> messages = torch.cat([messages, torch.zeros_like(messages)], dim=1)
    >>> round(Disent.posdis(attributes, messages), 6)
    0.978656

    Miniature language with perfect (=1) bosdis. Taken from Chaabouni et al. 2020, Appendix section 8.2.
    >>> attributes = torch.Tensor(
    ... [[0, 0], [0, 1], [0, 2], [0, 3],
    ... [1, 0], [1, 1], [1, 2], [1, 3],
    ... [2, 0], [2, 1], [2, 2], [2, 3],
    ... [3, 0], [3, 1], [3, 2], [3, 3]]
    ... )
    >>> messages = torch.Tensor(
    ... [[0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7],
    ... [1, 4, 1], [1, 5, 1], [1, 6, 1], [1, 7, 1],
    ... [2, 4, 2], [2, 5, 2], [2, 6, 2], [2, 7, 2],
    ... [3, 4, 3], [3, 3, 5], [3, 3, 6], [3, 3, 7]]
    ... )
    >>> Disent.bosdis(attributes, messages, vocab_size=3)
    1.0

    """

    def __init__(
        self,
        is_gumbel: bool,
        compute_posdis: bool = True,
        compute_bosdis: bool = False,
        vocab_size: int = 0,
        print_train: bool = False,
        print_test: bool = True,
    ):
        super().__init__()
        assert (
            print_train or print_test
        ), "At least one of `print_train` and `print_train` must be set"
        assert (
            compute_posdis or compute_bosdis
        ), "At least one of `compute_posdis` and `compute_bosdis` must be set"
        assert (
            not compute_bosdis or vocab_size > 0
        ), "To compute a positive vocab_size must be specifed"

        self.vocab_size = vocab_size
        self.is_gumbel = is_gumbel

        self.compute_posdis = compute_posdis
        self.compute_bosdis = compute_bosdis

        self.print_train = print_train
        self.print_test = print_test

    @staticmethod
    def bosdis(
        attributes: torch.Tensor, messages: torch.Tensor, vocab_size: int
    ) -> float:
        # print(f"message {messages.shape}")
        batch_size = messages.size(0)
        histogram = torch.zeros(batch_size, vocab_size, device=messages.device)
        for v in range(vocab_size):
            histogram[:, v] = messages.eq(v).sum(dim=-1)
        histogram = histogram[:, 1:]  # ignoring eos symbol
        return gap_mi_first_second(attributes, histogram)

    @staticmethod
    def posdis(attributes: torch.Tensor, messages: torch.Tensor) -> float:
        return gap_mi_first_second(attributes, messages)

