
def words_string(s):
    """
    This function splits a string of words separated by commas or spaces into an array of the words.

    Args:
        s (str): The input string to be split into words.

    Returns:
        list: A list of words in the string.

    Raises:
        TypeError: If the input is not a string.
        
    """
    
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    
    return s.replace(",", "").split()
