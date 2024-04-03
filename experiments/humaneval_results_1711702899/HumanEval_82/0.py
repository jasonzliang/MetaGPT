
def prime_length(string):
    """Check if the length of a string is a prime number.

    Args:
        string (str): The input string to check its length.

    Returns:
        bool: True if the length of the string is a prime number, False otherwise.

    Raises:
        TypeError: If the input parameter is not a string.
    """
    
    # Check if the input is a string
    if not isinstance(string, str):
        raise TypeError("The input must be a string.")
        
    # Get the length of the string
    n = len(string)
    
    # If the length is less than 2, it's not prime
    if n < 2:
        return False
    
    # Check for factors up to sqrt(n)
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
            
    return True
