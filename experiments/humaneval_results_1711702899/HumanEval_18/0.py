
def how_many_times(string: str, substring: str) -> int:
    """Find how many times a given substring can be found in the original string. Count overlaping cases.

    Args:
        string: The main string where we are looking for the substring.
        substring: The smaller string that we want to find in the main string.

    Returns:
        An integer representing how many times the substring appears in the string, including overlapping cases.

    Raises:
        TypeError: If either of the inputs is not a string.
    """
    # Check if both arguments are strings
    if not all(isinstance(i, str) for i in [string, substring]):
        raise TypeError("Both arguments must be strings.")
    
    return sum(1 for i in range(len(string)) if string.startswith(substring, i))
