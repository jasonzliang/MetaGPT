
def strlen(string: str) -> int:
    """Return length of given string.

    Args:
        string: The input string whose length is to be calculated.

    Returns:
        An integer representing the length of the input string.

    Raises:
        TypeError: If the input parameter is not a string.
    """
    if not isinstance(string, str):
        raise TypeError('The input must be a string')
    
    return len(string)
