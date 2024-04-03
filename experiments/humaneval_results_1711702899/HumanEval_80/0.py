
def is_happy(s):
    """Check if the string s is happy or not.

    A string is happy if its length is at least 3 and every 3 consecutive letters are distinct.

    Args:
        s: The input string to be checked.

    Returns:
        True if the string is happy, False otherwise.

    Raises:
        TypeError: If the argument is not a string.
    """
    
    # Check that s is a string
    if not isinstance(s, str):
        raise TypeError("The argument must be a string.")
        
    # Check length of s
    if len(s) < 3:
        return False
    
    # Check every 3 consecutive letters are distinct
    for i in range(len(s)-2):
        if len(set(s[i:i+3])) != 3:
            return False
        
    return True
