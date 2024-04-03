
def cycpattern_check(a , b):
    """Check if the second word or any of its rotations is a substring in the first word.

    Args:
        a (str): The first string to check for substrings.
        b (str): The second string which may be a rotation of the first string.

    Returns:
        bool: True if the second word or any of its rotations is a substring in the first word, False otherwise.

    Raises:
        TypeError: If either `a` or `b` are not strings.
    """
    
    # Check that both inputs are strings
    if not all(isinstance(i, str) for i in [a, b]):
        raise TypeError("Both arguments must be of type string.")
        
    # Generate all rotations of `b`
    rotations = [b[i:] + b[:i] for i in range(len(b))]
    
    # Check if any rotation is a substring of `a`
    return any(rotation in a for rotation in rotations)
