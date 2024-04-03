
def split_words(txt):
    """Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
    should split on commas ',' if no commas exists you should return the number of lower-case letters with odd order in the
    alphabet, ord('a') =0, ord('b') = 1, ... ord('z') = 25.

    Args:
        txt (str): The input string to be split into words.

    Returns:
        list or int: If whitespaces or commas exist in the text, a list of words is returned. Otherwise, the number of lower-case letters with odd order in the alphabet is returned.

    Raises:
        ValueError: If input is not a string.
    """
    
    if not isinstance(txt, str):
        raise ValueError("Input must be a string")
        
    # Split on whitespace
    words = txt.split()
    if len(words) > 1:
        return words
    
    # If no whitespaces exist, split on commas
    words = txt.split(',')
    if len(words) > 1:
        return words
        
    # Count lower-case letters with odd order in the alphabet
    count = 0
    for char in txt:
        if char.islower() and (ord(char) - ord('a')) % 2 == 1:
            count += 1
            
    return count
