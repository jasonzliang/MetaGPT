
from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """Separate groups of nested parentheses into separate strings.

    Args:
        paren_string: A string containing multiple groups of nested parentheses.

    Returns:
        A list of strings, where each string is a group of balanced and properly nested parentheses.

    Raises:
        ValueError: If the input string contains unbalanced or improperly nested parentheses.
    """
    paren_string = paren_string.replace(" ", "")  # Remove spaces from the string
    groups = []
    stack = []
    for char in paren_string:
        if char == '(':
            stack.append(char)
        elif char == ')':
            group = ''
            while stack and stack[-1] != '(':
                group = stack.pop() + group
            if not stack:  # If the stack is empty, there's a mismatching parenthesis
                raise ValueError("Unbalanced or improperly nested parentheses")
            stack.pop()  # Remove the opening bracket
            groups.append('(' + group + ')')
    if stack:  # If there are still items in the stack, there's a mismatching parenthesis
        raise ValueError("Unbalanced or improperly nested parentheses")
    return groups
