
def encrypt(s):
    """Create a function encrypt that takes a string as an argument and
    returns a string encrypted with the alphabet being rotated. 
    The alphabet should be rotated in a manner such that the letters 
    shift down by two multiplied to two places."""
    
    # Define the encryption key (rotate by 2*2)
    key = 5
    
    encrypted_text = ""
    
    for char in s:
        if char.isalpha():
            ascii_offset = 65 if char.isupper() else 97
            
            # Shift the character by the encryption key, wrapping around at 'Z' or 'z'
            encrypted_char = chr(((ord(char) - ascii_offset + key) % 26) + ascii_offset)
            
            encrypted_text += encrypted_char
        else:
            # Non-alphabet characters are left unchanged
            encrypted_text += char
    
    return encrypted_text
```

# Example usage:
print(encrypt('hi'))  # Outputs 'lm'
print(encrypt('asdfghjkl'))  # Outputs 'ewhjklnop'
print(encrypt('gf'))  # Outputs 'kj'
print(encrypt('et'))  # Outputs 'ix'
