def fix_spaces(text):
    text = list(text)
    i = 0
    while i < len(text):
        if text[i] == ' ':
            j = i + 1
            while j < len(text) and text[j] == ' ':
                j += 1
            if j - i > 2:
                text[i] = '-'
            else:
                text[i] = '_'
            del text[i+1:j]
        i += 1
    return ''.join(text)

print(fix_spaces("Example")) # "Example"
print(fix_spaces("Example 1")) # "Example_1"
print(fix_spaces(" Example 2")) # "_Example_2"
print(fix_spaces(" Example  3")) # "_Example-3" 