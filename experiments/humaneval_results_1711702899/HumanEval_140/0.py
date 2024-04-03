
def fix_spaces(text):
    """
    Given a string text, replace all spaces in it with underscores, 
    and if a string has more than 2 consecutive spaces, 
    then replace all consecutive spaces with - 

    Args:
        text (str): The input string.

    Returns:
        str: The modified string.

    Raises:
        TypeError: If the input is not a string.
        
    Examples:
        >>> fix_spaces("Example") 
        'Example'
        >>> fix_spaces("Example 1") 
        'Example_1'
        >>> fix_spaces(" Example 2") 
        '_Example_2'
        >>> fix_spaces(" Example  3") 
        '_Example-3'
    """
    
    if not isinstance(text, str):
        raise TypeError('Input must be a string')
    
    # Replace all spaces with underscores
    text = text.replace(' ', '_')
    
    # If there are more than 2 consecutive underscores, replace them with -
    while '___' in text:
        text = text.replace('___', '-')
        
    return text
