
def is_palindrome(text: str) -> bool:
    """
    Checks if given string is a palindrome.
    A palindrome is a word, phrase, number, or other sequence of characters 
    that reads the same forward and backward, ignoring spaces, punctuation, and capitalization.
    >>> is_palindrome('')
    True
    >>> is_palindrome('aba')
    True
    >>> is_palindrome('aaaaa')
    True
    >>> is_palindrome('zbcd')
    False
    """
    # Remove spaces and convert to lower case for comparison
    text = text.replace(" ", "").lower()

    # Compare the string with its reverse
    return text == text[::-1]
