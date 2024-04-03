
from typing import List

def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
    # Check if the list is empty
    if not numbers:
        raise ValueError("The input list cannot be empty")
        
    # Calculate mean of the list
    mean = sum(numbers) / len(numbers)
    
    # Calculate absolute deviation from mean for each number in the list
    abs_deviations = [abs(x - mean) for x in numbers]
    
    # Calculate and return Mean Absolute Deviation
    mad = sum(abs_deviations) / len(numbers)
    return mad
