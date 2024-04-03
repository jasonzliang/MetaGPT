
def is_palindrome(text: str) -> bool:
    """Checks if given string is a palindrome.

    Args:
        text: The string to check.

    Returns:
        True if the string is a palindrome, False otherwise.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(text, str):
        raise TypeError('Input must be a string')
    
    return text == text[::-1]
