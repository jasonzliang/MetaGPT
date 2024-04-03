
def sum_to_n(n: int) -> int:
    """sum_to_n is a function that sums numbers from 1 to n.

    Args:
        n (int): The number up to which the sum will be calculated.

    Returns:
        int: The sum of all integers from 1 to n.

    Raises:
        TypeError: If input is not an integer.
        ValueError: If input is less than 1.
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")
    elif n < 1:
        raise ValueError("Input must be greater than or equal to 1.")
    
    return sum(range(1, n + 1))
