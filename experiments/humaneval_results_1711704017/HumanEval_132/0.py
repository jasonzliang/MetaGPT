
def is_nested(string):
    """
    Function to check if a string of brackets has at least one nested subsequence.
    """
    # Initialize stack and flag
    stack = []
    nested = False

    try:
        for char in string:
            # Push opening bracket onto the stack
            if char == '[':
                stack.append(char)
            elif char == ']':
                # If there is a matching pair, pop it off the stack and set flag to True
                if len(stack) > 0:
                    nested = True
                    stack.pop()
                else:
                    return False
        # If the stack is not empty after scanning all characters, return False
        if len(stack) != 0:
            return False
    except Exception as e:
        print("An error occurred: ", str(e))
        return False
    
    # Return True only if a nested subsequence was found
    return nested
