
def decimal_to_binary(decimal):
    """Converts a given decimal number to binary format.

    Args:
        decimal (int): The decimal number to be converted.

    Returns:
        str: A string representing the binary equivalent of the input decimal, 
             with 'db' at the beginning and end.

    Raises:
        TypeError: If the input is not an integer.
    """
    
    if not isinstance(decimal, int):
        raise TypeError("The input must be an integer.")
        
    binary = bin(decimal).replace("0b", "")
    return 'db' + binary + 'db'
