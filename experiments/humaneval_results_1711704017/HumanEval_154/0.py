
def cycpattern_check(a , b):
    """You are given 2 words. You need to return True if the second word or any of its rotations is a substring in the first word"""
    # Check if either string is empty
    if not a or not b:
        return False
    
    # Generate all possible rotations of b
    for i in range(len(b)):
        rotation = b[i:] + b[:i]
        
        # If the rotation is a substring of a, return True
        if rotation in a:
            return True
    
    # If no rotation was found, return False
    return False
