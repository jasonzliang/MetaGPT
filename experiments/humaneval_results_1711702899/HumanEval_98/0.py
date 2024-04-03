
def count_upper(s):
    """
    Given a string s, count the number of uppercase vowels in even indices.

    Args:
        s (str): The input string.

    Returns:
        int: The number of uppercase vowels at even indices.

    Raises:
        TypeError: If the argument is not a string.
    """
    if not isinstance(s, str):
        raise TypeError('The argument must be a string')
    
    count = 0
    for i in range(len(s)):
        # Check if index is even and character is an uppercase vowel
        if i % 2 == 0 and s[i] in 'AEIOU':
            count += 1
            
    return count
