from language_analysis import Disent, TopographicSimilarity
import numpy as np
import pickle
import os
import torch
# attributes = torch.Tensor(
#     [[0,0], [0,1], [0,2], [0,3],
#     [1,0], [1,1], [1,2], [1,3],
#     [2,0], [2,1], [2,2], [2,3]]
# )
# case1: Perfect topsim and posdis but realtively low bosdis

# messages = attributes
# topsim 1.0
# posdis: 1.000000238418579
# bosdis: 0.3871784508228302
if __name__ == "__main__":
    # [color, shape]
    attributes = torch.Tensor(
        [[0,0], [0,1], [0,2], [0,3], # [red, triangle], [red, square], [red circle], [red star]
        [1,0], [1,1], [1,2], [1,3],  # [green, triangle], [green, square], [green circle], [green star]
        [2,0], [2,1], [2,2], [2,3]]  # [blue, triangle], [blue, square], [blue circle], [blue star]
    )

    # messages1: Perfect topsim and posdis, but realtively low bosdis
    messages1 = attributes

    # each combination has different lengths (with zero padding)
    messages2 = torch.Tensor(
        [[4,5,1,0,0,0,0], [4,5,8,9,7,8,0], [4,5,9,4,6,3,0], [4,5,2,7,3,0,0], # 4,5 means red
        [7,1,0,0,0,0,0], [7,8,9,7,8,0,0], [7,9,4,6,3,0,0], [7,2,7,3,0,0,0], # 7 means green
        [9,6,2,1,0,0,0], [9,6,2,8,9,7,8], [9,6,2,9,4,6,3], [9,6,2,2,7,3,0] # 9,6,2 means blue
        ]
    )

    # each combination has different lengths (with zero padding)
    messages3 = torch.Tensor(
        [[1, 8,9,3,2,5, 0,0,0], [1, 7,4,6,8, 0,0,0,0], [1, 3,2, 0,0,0,0,0,0], [1, 8, 0,0,0,0,0,0,0],
        [4,5,7, 8,9,3,2,5, 0], [4,5,7, 7,4,6,8, 0,0], [4,5,7, 3,2, 0,0,0,0], [4,5,7, 8, 0,0,0,0,0],
        [8,6,7,9, 8,9,3,2,5], [8,6,7,9, 7,4,6,8, 0], [8,6,7,9, 3,2, 0,0,0], [8,6,7,9, 8, 0,0,0,0]
        ]
    )

    # same as messages3 but segment occurs in random positions
    messages4 = torch.Tensor(
        [[1, 8,9,3,2,5, 0,0,0], [0,0,0,0, 1, 7,4,6,8], [1, 3,2, 0,0,0,0,0,0], [1, 8, 0,0,0,0,0,0,0],
        [4,5,7, 8,9,3,2,5, 0], [4,5,7, 7,4,6,8, 0,0], [0,0,0,0, 4,5,7, 3,2], [4,5,7, 8, 0,0,0,0,0],
        [8,6,7,9, 8,9,3,2,5], [8,6,7,9, 7,4,6,8, 0], [0,0, 8,6,7,9, 3,2, 0], [8,6,7,9, 8, 0,0,0,0]
        ]
    )

    topsim = TopographicSimilarity.compute_topsim(attributes, attributes)
    posdis = Disent.posdis(attributes, attributes)
    bosdis = Disent.bosdis(attributes, attributes, vocab_size=16)
    print("case1: Perfect topsim and posdis but realtively low bosdis")
    print(f"messages=attributes")
    print(f"topsim {topsim}, posdis: {posdis}, bosdis: {bosdis}")   

    topsim = TopographicSimilarity.compute_topsim(attributes, messages2)
    posdis = Disent.posdis(attributes, messages2)
    bosdis = Disent.bosdis(attributes, messages2, vocab_size=16)
    print("case2: Each word has different length resulting in low posdis")
    print(f"topsim {topsim}, posdis: {posdis}, bosdis: {bosdis}")   


    topsim = TopographicSimilarity.compute_topsim(attributes, messages3)
    posdis = Disent.posdis(attributes, messages3)
    bosdis = Disent.bosdis(attributes, messages3, vocab_size=16)
    print("case3:")
    print(f"topsim {topsim}, posdis: {posdis}, bosdis: {bosdis}")   


    topsim = TopographicSimilarity.compute_topsim(attributes, messages4)
    posdis = Disent.posdis(attributes, messages4)
    bosdis = Disent.bosdis(attributes, messages4, vocab_size=16)
    print("case4:")
    print(f"topsim {topsim}, posdis: {posdis}, bosdis: {bosdis}")