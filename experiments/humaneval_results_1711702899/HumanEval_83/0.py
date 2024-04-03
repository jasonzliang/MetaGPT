
def starts_one_ends(n):
    """
    Given a positive integer n, return the count of the numbers of n-digit
    positive integers that start or end with 1.

    Args:
        n (int): The number of digits in the positive integers.

    Returns:
        int: The count of the numbers of n-digit positive integers that start or end with 1.

    Raises:
        ValueError: If the input is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("The input must be a positive integer.")
    
    return 2 * (10 ** (n-1)) - 10 ** (n//2 - 1)
