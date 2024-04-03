
import hashlib

def string_to_md5(text):
    if text == "":
        return None
    else:
        result = hashlib.md5(text.encode())
        return result.hexdigest()
    