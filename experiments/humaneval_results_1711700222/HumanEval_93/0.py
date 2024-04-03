
def encode(message):
    vowels = 'aeiouAEIOU'
    encoded_msg = ''
    for char in message:
        if char.isalpha():
            if char in vowels:
                index = (vowels.index(char) + 2) % 5
                if char.isupper():
                    encoded_msg += vowels[index].upper()
                else:
                    encoded_msg += vowels[index]
            elif char.isupper():
                encoded_msg += char.lower()
            else:
                encoded_msg += char.upper()
        else:
            encoded_msg += char
    return encoded_msg
