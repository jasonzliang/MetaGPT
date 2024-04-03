
def string_xor(a: str, b: str) -> str:
    """Perform binary XOR on two strings of 1s and 0s.

    Args:
        a: The first string consisting only of 1s and 0s.
        b: The second string consisting only of 1s and 0s.

    Returns:
        A string that is the result of performing XOR on the two input strings.

    Raises:
        ValueError: If the input strings are not valid binary strings (i.e., they contain characters other than '1' or '0').
    """
    
    # Check if inputs are valid binary strings
    for s in [a, b]:
        if any(c != '0' and c != '1' for c in s):
            raise ValueError('Inputs must be binary strings (containing only 1s and 0s)')
    
    # Perform XOR operation on the two input strings
    result = ''
    for i in range(len(a)):
        if a[i] == b[i]:
            result += '0'
        else:
            result += '1'
            
    return result
