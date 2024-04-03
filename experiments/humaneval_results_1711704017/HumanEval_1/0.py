
def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    # Remove all whitespace from the string
    paren_string = paren_string.replace(" ", "")

    groups = []
    start = 0
    count = 0

    for i, char in enumerate(paren_string):
        if char == '(':
            # If this is the first parenthesis in a group, remember its position
            if count == 0:
                start = i
            count += 1
        elif char == ')':
            count -= 1
            # If we have closed all parentheses in this group, add it to our list
            if count == 0:
                groups.append(paren_string[start:i+1])

    return groups
