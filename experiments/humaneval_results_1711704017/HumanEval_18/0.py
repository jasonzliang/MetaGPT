
def how_many_times(string: str, substring: str) -> int:
    """ Find how many times a given substring can be found in the original string. Count overlaping cases.
    >>> how_many_times('', 'a')
    0
    >>> how_many_times('aaa', 'a')
    3
    >>> how_many_times('aaaa', 'aa')
    3
    """
    # Check if the inputs are valid
    if not isinstance(string, str) or not isinstance(substring, str):
        raise ValueError("Both arguments must be strings.")
        
    # If string or substring is empty, return 0
    if len(string) == 0 or len(substring) == 0:
        return 0
    
    count = start = 0
    while start < len(string):
        pos = string.find(substring, start)
        
        # If substring is not found in the remaining part of the string
        if pos != -1:
            count += 1
            start = pos + 1  # Move to next position after found substring
        else:
            break
    
    return count
