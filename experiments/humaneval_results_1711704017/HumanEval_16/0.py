
def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct characters (regardless of case) does it consist of.
    >>> count_distinct_characters('xyzXYZ')
    3
    >>> count_distinct_characters('Jerry')
    4
    """
    # Error handling: if the input is not a string, raise an error
    if not isinstance(string, str):
        raise ValueError("Input must be a string")
        
    # Convert to lowercase and use set to remove duplicates, then return length of set
    distinct_chars = set(string.lower())
    
    return len(distinct_chars)
