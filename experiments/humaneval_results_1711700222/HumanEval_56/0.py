
def correct_bracketing(brackets):
    count = 0
    for bracket in brackets:
        if bracket == '<':
            count += 1
        elif bracket == '>':
            if count == 0:
                return False
            count -= 1
    return count == 0
