
def correct_bracketing(brackets: str) -> bool:
    """Check if every opening bracket has a corresponding closing bracket.

    Args:
        brackets: A string of "<" and ">".

    Returns:
        True if every opening bracket has a corresponding closing bracket, False otherwise.

    Raises:
        TypeError: If the input is not a string.
    """
    # Check that the input is a string
    if not isinstance(brackets, str):
        raise TypeError("Input must be a string")
    
    # Initialize counter for open brackets
    open_count = 0
    
    # Iterate over each character in the string
    for char in brackets:
        # If the character is an opening bracket, increment the counter
        if char == "<":
            open_count += 1
        # If the character is a closing bracket, decrement the counter
        elif char == ">":
            open_count -= 1
        
        # If at any point the counter goes negative, return False
        if open_count < 0:
            return False
    
    # If the counter is 0 after iterating over all characters, return True
    # Otherwise, return False
    return open_count == 0
