
def encode(message):
    vowels = 'aeiouAEIOU'
    encoded_msg = ''
    for char in message:
        if char.isalpha():
            # Swap case and replace vowel
            new_char = char.swapcase()
            if new_char in vowels:
                index = (vowels.index(new_char) + 2) % 26
                new_char = vowels[index]
            encoded_msg += new_char
        else:
            # Non-letter characters are kept as is
            encoded_msg += char
    return encoded_msg
