
def check_if_last_char_is_a_letter(txt):
    if not txt or not txt[-1].isalpha():
        return False
    else:
        words = txt.split()
        last_word = words[-1]
        return last_word[-1] == txt[-1]
    