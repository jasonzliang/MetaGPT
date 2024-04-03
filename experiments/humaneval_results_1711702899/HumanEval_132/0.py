
def is_nested(brackets):
    """This function takes a string of brackets as input and returns True if there is at least one nested bracket in the sequence.

    Args:
        brackets (str): A string containing only square brackets.

    Returns:
        bool: True if there is at least one valid subsequence with a nested bracket, False otherwise.

    Raises:
        ValueError: If the input string contains any character other than square brackets.
    """
    
    # Check for invalid characters in the string
    if not all(char in '[]' for char in brackets):
        raise ValueError("Input string can only contain square brackets.")
        
    stack = []
    for bracket in brackets:
        if bracket == '[':
            # Push opening brackets onto the stack
            stack.append(bracket)
        elif bracket == ']':
            # If there is a closing bracket and no opening brackets on the stack, return False
            if not stack:
                return False
            else:
                # Pop the last opening bracket from the stack
                stack.pop()
                
    # If there are any remaining opening brackets on the stack, return True
    return bool(stack)
