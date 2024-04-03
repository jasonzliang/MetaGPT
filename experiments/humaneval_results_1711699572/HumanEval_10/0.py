def make_palindrome(string: str) -> str:
    for i in range(len(string), 0, -1):
        if is_palindrome(string[:i]):
            return string + string[i-1::-1]
    return string + string[::-1] 