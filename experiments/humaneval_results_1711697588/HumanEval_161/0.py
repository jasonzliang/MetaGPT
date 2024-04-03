
def solve(s):
    if any(char.isalpha() for char in s):
        return ''.join([char.lower() if char.isupper() else char.upper() for char in s])
    else:
        return s[::-1]
    