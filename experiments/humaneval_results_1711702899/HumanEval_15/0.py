
def string_sequence(n: int) -> str:
    """Return a string containing space-delimited numbers starting from 0 upto n inclusive.

    Args:
        n (int): The upper limit of the sequence.

    Returns:
        str: A string with space-separated numbers from 0 to n inclusive.

    Raises:
        TypeError: If input is not an integer.
        ValueError: If input is negative.
    """
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")
    elif n < 0:
        raise ValueError("Input must be a non-negative integer.")
    
    return ' '.join(str(i) for i in range(n + 1))
