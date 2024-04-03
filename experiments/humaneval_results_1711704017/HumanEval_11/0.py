
def string_xor(a: str, b: str) -> str:
    """ Input are two strings a and b consisting only of 1s and 0s.
    Perform binary XOR on these inputs and return result also as a string.
    >>> string_xor('010', '110')
    '100'
    """
    # Check if the strings are valid (contain only 1s and 0s)
    for char in a:
        if char not in ['0', '1']:
            raise ValueError("Invalid input string a. It should contain only 1s and 0s.")
    for char in b:
        if char not in ['0', '1']:
            raise ValueError("Invalid input string b. It should contain only 1s and 0s.")
    
    # Check if the strings have equal length
    if len(a) != len(b):
        raise ValueError("Strings a and b must be of equal length for XOR operation.")
    
    # Perform binary XOR on each pair of corresponding characters in the strings
    result = ''
    for i in range(len(a)):
        if a[i] == b[i]:
            result += '0'
        else:
            result += '1'
    
    return result
