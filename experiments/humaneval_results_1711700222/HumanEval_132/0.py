
def is_nested(string):
    stack = []
    for char in string:
        if char == '[':
            stack.append(char)
        elif char == ']':
            if not stack or (stack and stack[-1] != '['):
                return True
            else:
                stack.pop()
    return False
