
def flip_case(string: str) -> str:
    """For a given string, flip lowercase characters to uppercase and uppercase to lowercase.

    Args:
        string: The input string that needs to be flipped.

    Returns:
        A new string with all the case of characters flipped.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(string, str):
        raise TypeError('The input must be a string')
    
    return ''.join([char.upper() if char.islower() else char.lower() for char in string])
