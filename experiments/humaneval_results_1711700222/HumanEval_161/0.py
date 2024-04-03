
def solve(s):
    if any(char.isalpha() for char in s):
        return ''.join([char.upper() if char.islower() else char.lower() for char in s])
    else:
        return s[::-1]
