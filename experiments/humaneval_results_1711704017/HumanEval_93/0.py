
def encode(message):
    """
    Encode a message by swapping case of all letters and replacing vowels with the letter that appears 2 places ahead.
    Assume only letters.
    """
    # Define the mapping for vowel replacement
    vowel_map = {'a': 'c', 'e': 'g', 'i': 'k', 'o': 'm', 'u': 's',
                 'A': 'C', 'E': 'G', 'I': 'K', 'O': 'M', 'U': 'S'}
    
    # Initialize an empty result string
    encoded_message = ''
    
    for char in message:
        if char.isalpha():  # If the character is a letter
            if char in vowel_map:  # If it's a vowel, replace it
                encoded_char = vowel_map[char]
            else:  # Otherwise, swap its case
                encoded_char = char.lower() if char.isupper() else char.upper()
        else:  # Non-letter characters remain the same
            encoded_char = char
        
        encoded_message += encoded_char
    
    return encoded_message
