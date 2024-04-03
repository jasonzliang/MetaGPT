
def encode(message):
    """Encodes a message by swapping case of all letters and replacing vowels 
    with the letter that appears two places ahead in the English alphabet.

    Args:
        message (str): The message to be encoded.

    Returns:
        str: The encoded message.

    Raises:
        TypeError: If the input is not a string.
    """
    
    if not isinstance(message, str):
        raise TypeError('Input must be a string')
        
    vowels = 'aeiouAEIOU'
    encoded_msg = ''
    
    for char in message:
        # Swap case of letters and replace vowels
        if char.isalpha():
            if char in vowels:
                new_char = chr((ord(char) - ord('a') + 2) % 26 + ord('a')) if char.islower() else chr((ord(char) - ord('A') + 2) % 26 + ord('A'))
            else:
                new_char = char.swapcase()
        # Keep non-letter characters as they are
        else:
            new_char = char
            
        encoded_msg += new_char
        
    return encoded_msg
