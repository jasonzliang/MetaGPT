
def count_distinct_characters(string: str) -> int:
    """Given a string, find out how many distinct characters (regardless of case) does it consist of.

    Args:
        string: The input string from which to count the number of distinct characters.

    Returns:
        An integer representing the number of distinct characters in the input string.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(string, str):
        raise TypeError('The input must be a string')
    
    return len(set(string.lower()))
