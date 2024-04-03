
def flip_case(string: str) -> str:
    """ For a given string, flip lowercase characters to uppercase and uppercase to lowercase.
    >>> flip_case('Hello')
    'hELLO'
    """
    # Check if the input is a string
    if not isinstance(string, str):
        raise ValueError("Input must be a string")
        
    flipped = ''
    for char in string:
        # If character is uppercase, convert to lowercase and vice versa
        if char.isupper():
            flipped += char.lower()
        elif char.islower():
            flipped += char.upper()
        else:
            # Keep non-alphabet characters as they are
            flipped += char
    return flipped
