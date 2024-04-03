
def parse_nested_parens(paren_string: str) -> List[int]:
    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.
    For each of the group, output the deepest level of nesting of parentheses.
    E.g. (()()) has maximum two levels of nesting while ((())) has three.

    >>> parse_nested_parens('(()()) ((())) () ((())()())')
    [2, 3, 1, 3]
    """
    def count_depth(s: str) -> int:
        """Helper function to calculate the maximum depth of nested parentheses in a string."""
        max_depth = current_depth = 0
        for char in s:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth
    
    # Split the input string into groups and calculate depth for each group
    paren_groups = paren_string.split()
    depths = [count_depth(group) for group in paren_groups]
    
    return depths
