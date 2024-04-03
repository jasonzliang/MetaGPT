
from typing import List, Tuple
import functools
import operator

def sum_product(numbers: List[int]) -> Tuple[int, int]:
    if not numbers:
        return 0, 1
    else:
        return sum(numbers), functools.reduce(operator.mul, numbers, 1)
