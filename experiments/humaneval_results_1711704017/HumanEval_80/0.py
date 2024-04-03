
def is_happy(s):
    """Check if the string s is happy. A string is happy if its length is at least 3 and every 3 consecutive letters are distinct."""
    # Check if the string's length is less than 3, return False immediately
    if len(s) < 3:
        return False
    
    # Iterate over the string with a step of 3
    for i in range(len(s)-2):
        # If any three consecutive letters are not distinct, return False
        if len(set(s[i:i+3])) != 3:
            return False
    
    # If no three consecutive letters were found to be not distinct, the string is happy
    return True
