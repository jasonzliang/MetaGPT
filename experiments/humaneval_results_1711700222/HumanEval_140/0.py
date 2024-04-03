
import re

def fix_spaces(text):
    text = text.strip()  # Remove leading and trailing spaces
    text = re.sub(' +', ' ', text)  # Replace multiple spaces with single space
    if '  ' in text:
        return text.replace(' ', '-').replace('_', ' ')
    else:
        return text.replace(' ', '_')
