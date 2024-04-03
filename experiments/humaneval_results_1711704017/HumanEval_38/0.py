
def encode_cyclic(s: str) -> str:
    """
    Returns encoded string by cycling groups of three characters.
    """
    # Split the string into groups of 3 characters each.
    groups = [s[i:i+3] for i in range(0, len(s), 3)]
    
    # Cycle elements in each group unless it has fewer than 3 characters.
    encoded_groups = [group[1:] + group[0] if len(group) == 3 else group for group in groups]
    
    return "".join(encoded_groups)

def decode_cyclic(s: str) -> str:
    """
    Takes as input a string encoded with encode_cyclic function. Returns the decoded string.
    """
    # Split the string into groups of 3 characters each.
    groups = [s[i:i+3] for i in range(0, len(s), 3)]
    
    # Decycle elements in each group unless it has fewer than 3 characters.
    decoded_groups = [group[-1] + group[:-1] if len(group) == 3 else group for group in groups]
    
    return "".join(decoded_groups)
