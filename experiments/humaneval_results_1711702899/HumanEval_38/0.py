
def encode_cyclic(s: str) -> str:
    """Encodes a string by cycling groups of three characters.

    Args:
        s: The string to be encoded.

    Returns:
        The encoded string.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    
    # split string to groups. Each of length 3.
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    
    # cycle elements in each group. Unless group has fewer elements than 3.
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]
    
    return "".join(groups)

def decode_cyclic(s: str) -> str:
    """Decodes a string that was encoded with the encode_cyclic function.

    Args:
        s: The string to be decoded.

    Returns:
        The decoded string.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    
    # split string to groups. Each of length 3.
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]
    
    # reverse cycle elements in each group. Unless group has fewer elements than 3.
    groups = [(group[-1] + group[:-1]) if len(group) == 3 else group for group in groups]
    
    return "".join(groups)
