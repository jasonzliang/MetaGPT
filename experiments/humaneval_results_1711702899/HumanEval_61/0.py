
def correct_bracketing(brackets: str) -> bool:
    """Check if every opening bracket has a corresponding closing bracket.

    Args:
        brackets: A string of "(" and ")".

    Returns:
        True if every opening bracket has a corresponding closing bracket, False otherwise.

    Raises:
        TypeError: If the input is not a string.
    """
    # Check that the input is a string
    if not isinstance(brackets, str):
        raise TypeError("Input must be a string")
    
    # Initialize a counter for open brackets
    count = 0
    
    # Iterate over each character in the string
    for char in brackets:
        # If the character is an opening bracket, increment the counter
        if char == "(":
            count += 1
        # If the character is a closing bracket, decrement the counter
        elif char == ")":
            count -= 1
        
        # If the counter goes negative, there's a closing bracket without a corresponding opening bracket
        if count < 0:
            return False
    
    # If the counter is not zero at the end, there are unmatched opening brackets
    return count == 0
