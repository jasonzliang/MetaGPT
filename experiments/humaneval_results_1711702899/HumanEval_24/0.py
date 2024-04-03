
def largest_divisor(n: int) -> int:
    """For a given number n, find the largest number that divides n evenly, smaller than n.

    Args:
        n: The input integer for which we need to find the largest divisor.

    Returns:
        The largest divisor of n that is less than n.

    Raises:
        ValueError: If n is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    for i in range(n-1, 0, -1):
        if n % i == 0:
            return i
