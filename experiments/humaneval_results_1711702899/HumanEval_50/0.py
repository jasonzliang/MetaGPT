
def encode_shift(s: str) -> str:
    """Encodes a string by shifting every character by 5 in the alphabet.

    Args:
        s: The string to be encoded.

    Returns:
        The encoded string.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    
    return "".join([chr(((ord(ch) - ord('a') + 5) % 26) + ord('a')) for ch in s])

def decode_shift(s: str) -> str:
    """Decodes a string that was encoded with the encode_shift function.

    Args:
        s: The string to be decoded.

    Returns:
        The decoded string.

    Raises:
        TypeError: If the input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    
    return "".join([chr(((ord(ch) - ord('a') - 5) % 26) + ord('a')) for ch in s])
