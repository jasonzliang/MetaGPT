
from typing import List

def mean_absolute_deviation(numbers: List[float]) -> float:
    """Calculate Mean Absolute Deviation around the mean of a dataset.

    Args:
        numbers: A list of floats for which to calculate MAD.

    Returns:
        The Mean Absolute Deviation (MAD) of the input data.

    Raises:
        ValueError: If the input list is empty.
    """
    if not numbers:
        raise ValueError("Input list cannot be empty")
    
    mean = sum(numbers) / len(numbers)
    absolute_deviations = [abs(x - mean) for x in numbers]
    mad = sum(absolute_deviations) / len(numbers)
    
    return mad
