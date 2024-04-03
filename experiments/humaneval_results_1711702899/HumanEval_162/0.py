
import hashlib

def string_to_md5(text):
    """
    Given a string 'text', return its md5 hash equivalent string.
    If 'text' is an empty string, return None.

    Args:
        text (str): The input string to be hashed.

    Returns:
        str or None: The MD5 hash of the input string if it's not an empty string, otherwise None.

    Raises:
        TypeError: If 'text' is not a string.
    """
    
    # Checking if text is a string
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
        
    # Return None for empty strings
    if text == "":
        return None
    
    # Calculate and return the MD5 hash of the input string
    return hashlib.md5(text.encode()).hexdigest()
