
def shift_char(ch: str, shift: int) -> str:
    """
    Shifts a character by the given amount.
    Only works for lowercase alphabets.
    """
    if not ch.isalpha() or not ch.islower():
        raise ValueError("Input must be a lowercase letter.")
    
    # Calculate new ASCII value, wrapping around within the range of lowercase letters
    ascii_val = ord(ch) + shift
    if ascii_val > ord('z'):
        ascii_val -= 26
    elif ascii_val < ord('a'):
        ascii_val += 26
    
    return chr(ascii_val)

def encode_shift(s: str, shift: int = 5) -> str:
    """
    Returns encoded string by shifting every character by the given amount.
    Only works for lowercase alphabets.
    """
    try:
        return "".join([shift_char(ch, shift) for ch in s])
    except ValueError as e:
        print("Invalid input:", str(e))
        return None

def decode_shift(s: str, shift: int = 5) -> str:
    """
    Takes as input string encoded with encode_shift function. 
    Returns decoded string by shifting every character back by the given amount.
    Only works for lowercase alphabets.
    """
    try:
        return "".join([shift_char(ch, -shift) for ch in s])
    except ValueError as e:
        print("Invalid input:", str(e))
        return None
