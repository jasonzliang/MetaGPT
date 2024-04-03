
def make_palindrome(string: str) -> str:
    """ Find the shortest palindrome that begins with a supplied string.

    Args:
        string: The input string to be transformed into a palindrome.

    Returns:
        The shortest palindrome that begins with the supplied string.

    Raises:
        TypeError: If the input is not a string.
        
    """
    
    if not isinstance(string, str):
        raise TypeError('Input must be a string')
    
    # Find the longest postfix of supplied string that is a palindrome.
    for i in range(len(string), -1, -1):
        suffix = string[i:]
        
        if is_palindrome(suffix):
            prefix = string[:i]
            
            # Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
            return string + prefix[::-1]
