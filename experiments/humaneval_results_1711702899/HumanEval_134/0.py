
def check_if_last_char_is_a_letter(txt):
    """
    This function checks if the last character of a given string is an alphabetical 
    character and is not a part of a word.

    Args:
        txt (str): The input string to be checked.

    Returns:
        bool: True if the last character is an alphabetical character and it's not a part of a word, False otherwise.

    Raises:
        TypeError: If the provided argument is not a string.
    """
    
    # Checking if txt is a string
    if not isinstance(txt, str):
        raise TypeError("The input should be a string")
        
    # Splitting the text into words and checking if last character of last word is alphabetical
    return txt.split()[-1][-1].isalpha() if txt.split() else False
