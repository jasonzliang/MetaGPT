
def check_if_last_char_is_a_letter(txt):
    '''
    Create a function that returns True if the last character
    of a given string is an alphabetical character and is not
    a part of a word, and False otherwise.
    Note: "word" is a group of characters separated by space.

    Examples:
    check_if_last_char_is_a_letter("apple pie") ➞ False
    check_if_last_char_is_a_letter("apple pi e") ➞ True
    check_if_last_char_is_a_letter("apple pi e ") ➞ False
    check_if_last_char_is_a_letter("") ➞ False 
    '''
    
    # Check if the string is empty
    if not txt:
        return False

    # Split the text into words
    words = txt.split()

    # If there are no words, check the last character of the entire string
    if len(words) == 0:
        return txt[-1].isalpha() and txt[-2] == ' '
    
    # Check the last character of the last word
    last_word = words[-1]
    return last_word[-1].isalpha() and (len(txt) == len(last_word) or txt[-len(last_word)-1] == ' ')
