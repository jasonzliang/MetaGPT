
def strlen(string: str) -> int:
    """Return length of given string.

    Args:
        string (str): The input string.

    Returns:
        int: Length of the string.

    Raises:
        TypeError: If the input is not a string.

    Examples:
        >>> strlen('')
        0
        >>> strlen('abc')
        3
    """
    if not isinstance(string, str):
        raise TypeError("Input must be a string")
    
    length = 0
    for _ in string:
        length += 1
    return length
