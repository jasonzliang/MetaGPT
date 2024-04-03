
def correct_bracketing(brackets: str) -> bool:
    """Check if every opening bracket has a corresponding closing bracket."""
    count = 0
    for b in brackets:
        if b == '<':
            count += 1
        elif b == '>':
            if count == 0:
                return False
            count -= 1
        else:
            raise ValueError("Invalid character in input string")
    return count == 0
