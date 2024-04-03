
def digitSum(s):
    """Calculate the sum of ASCII codes for uppercase characters in a string.

    Args:
        s (str): The input string.

    Returns:
        int: The sum of ASCII codes for uppercase characters in the string.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    
    return sum(ord(c) for c in s if c.isupper())
