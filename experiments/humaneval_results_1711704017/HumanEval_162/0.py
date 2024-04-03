
import hashlib

def string_to_md5(text):
    """
    Given a string 'text', return its md5 hash equivalent string.
    If 'text' is an empty string, return None.
    """
    if text == "":
        return None
    else:
        # Create a new md5 hash object
        hash_object = hashlib.md5()
        
        # Update the hash object with the bytes of the input string
        hash_object.update(text.encode())
        
        # Return the hexadecimal digest of the hash object
        return hash_object.hexdigest()
