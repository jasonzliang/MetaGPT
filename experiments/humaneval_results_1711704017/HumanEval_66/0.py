
def digitSum(s):
    """
    Function to calculate the sum of ASCII values of uppercase characters in a string.

    Args:
        s (str): The input string.

    Returns:
        int: The sum of ASCII values of uppercase characters.
    """
    # Initialize sum as 0
    total = 0
    
    # Iterate over each character in the string
    for char in s:
        # If the character is an uppercase letter, add its ASCII value to the sum
        if char.isupper():
            total += ord(char)
            
    return total
