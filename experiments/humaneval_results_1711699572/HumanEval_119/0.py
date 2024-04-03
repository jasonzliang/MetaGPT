
def match_parens(lst):
    # Concatenate the two strings in some order.
    s = lst[0] + lst[1]
    stack = []
    for c in s:
        if c == '(':
            stack.append(c)
        elif c == ')':
            # If there is no matching opening parenthesis, return 'No'.
            if not stack:
                return 'No'
            else:
                stack.pop()
    # If all parentheses are matched, return 'Yes'.
    if not stack:
        return 'Yes'
    else:
        return 'No'
