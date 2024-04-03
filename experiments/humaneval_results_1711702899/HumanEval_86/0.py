
def anti_shuffle(s):
    """
    This function takes a string and returns an ordered version of it. 
    Ordered version of string is a string where all words (separated by space) are replaced by a new word where all the characters arranged in ascending order based on ascii value.
    
    Args:
        s: The input string to be anti-shuffled.
        
    Returns:
        A string with each word sorted in ascending order based on ASCII values.
        
    Raises:
        TypeError: If the input is not a string.
    
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    
    words = s.split(' ')
    sorted_words = [''.join(sorted(word)) for word in words]
    return ' '.join(sorted_words)
