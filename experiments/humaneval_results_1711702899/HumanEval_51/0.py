
def remove_vowels(text):
    """
    This function takes a string as input and returns the same string without vowels.

    Args:
        text (str): The string from which to remove vowels.

    Returns:
        str: The input string with all vowels removed.

    Raises:
        TypeError: If the argument is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("The argument must be a string.")
    
    return "".join([char for char in text if char.lower() not in 'aeiou'])
