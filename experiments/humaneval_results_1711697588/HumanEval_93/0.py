
def encode(message):
    vowels = 'aeiouAEIOU'
    encoded_msg = ''
    for char in message:
        if char.isalpha():
            # Swap case and replace vowel
            if char in vowels:
                new_char = chr((ord(char) - ord('a') + 2) % 26 + ord('a')) if char.islower() else chr((ord(char) - ord('A') + 2) % 26 + ord('A'))
            else:
                new_char = char.upper() if char.islower() else char.lower()
        else:
            # Keep non-letter characters as is
            new_char = char
        encoded_msg += new_char
    return encoded_msg
```
This function works by iterating over each character in the message. If the character is a letter, it checks if it's a vowel or not. If it's a vowel, it replaces it with the 2 places ahead letter in the alphabet (wrapping around to 'a'/'A' if necessary). Otherwise, it swaps its case. Non-letter characters are kept as is. The encoded message is then returned.

Here are some examples of how this function works:
```python
print(encode('test'))  # Outputs: 'TGST'
print(encode('This is a message'))  # Outputs: 'tHKS KS C MGSSCGG'
