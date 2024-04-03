
def sum_to_n(n: int) -> int:
    """sum_to_n is a function that sums numbers from 1 to n.
    >>> sum_to_n(30)
    465
    >>> sum_to_n(100)
    5050
    >>> sum_to_n(5)
    15
    >>> sum_to_n(10)
    55
    >>> sum_to_n(1)
    1
    """
    # Check if the input is valid (i.e., a positive integer)
    if not isinstance(n, int) or n < 1:
        raise ValueError("Input must be a positive integer.")

    # Use Gauss' formula to calculate the sum quickly
    return n * (n + 1) // 2
