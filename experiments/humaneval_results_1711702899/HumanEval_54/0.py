
def same_chars(s0: str, s1: str) -> bool:
    """Check if two words have the same characters.

    Args:
        s0: The first word to compare.
        s1: The second word to compare.

    Returns:
        True if both words contain exactly the same characters, False otherwise.

    Raises:
        TypeError: If either of the inputs is not a string.
    """
    # Check that both inputs are strings
    if not all(isinstance(s, str) for s in [s0, s1]):
        raise TypeError('Both arguments must be strings')
    
    return sorted(s0) == sorted(s1)
