
from typing import List
import numpy as np

def mean_absolute_deviation(numbers: List[float]) -> float:
    x = np.array(numbers)
    x_mean = np.mean(x)
    return np.mean(np.abs(x - x_mean))
