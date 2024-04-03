def correct_bracketing(brackets):
    count = 0
    for bracket in brackets:
        if bracket == '(':
            count += 1
        elif bracket == ')':
            count -= 1
        if count < 0:  # closing bracket before opening one
            return False
    return count == 0  # all brackets should be closed