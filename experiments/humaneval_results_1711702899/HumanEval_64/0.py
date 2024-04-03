
def vowels_count(s):
    """Count the number of vowels in a string.

    Args:
        s (str): The input string.

    Returns:
        int: The number of vowels in the string.

    Raises:
        TypeError: If the argument is not a string.
    """
    
    if not isinstance(s, str):
        raise TypeError("The argument must be a string.")
        
    count = 0
    for char in s:
        if char.lower() in 'aeiou':
            count += 1
            
    # If the last character is 'y', also count it as a vowel
    if s[-1].lower() == 'y':
        count += 1
        
    return count
