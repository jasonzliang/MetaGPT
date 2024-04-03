
def string_sequence(n: int) -> str:
    """Return a string containing space-delimited numbers starting from 0 upto n inclusive.

    Args:
        n (int): The upper limit of the sequence.

    Returns:
        str: A string with space-separated numbers from 0 to n.

    Raises:
        ValueError: If input is not an integer or less than 0.
    """
    
    # Error handling for invalid inputs
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input must be a non-negative integer.")
        
    return ' '.join(str(i) for i in range(n + 1))
