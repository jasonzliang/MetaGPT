
def separate_paren_groups(paren_string):
    paren_string = paren_string.replace(" ", "")
    stack = []
    result = []
    temp = ""

    for char in paren_string:
        if char == '(':
            stack.append(char)
            temp += char
        elif char == ')':
            stack.pop()
            temp += char
            if not stack:
                result.append(temp)
                temp = ""
    return result
