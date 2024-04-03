
def match_parens(lst):
    """
    Check if it is possible to concatenate the two strings in some order, 
    that the resulting string will be good. A string S is considered to be good 
    if all parentheses in S are balanced.
    """
    # Combine both strings into one
    s = lst[0] + lst[1]
    
    # Initialize a counter for open and close parentheses
    open_count = 0
    close_count = 0
    
    # Iterate over the string
    for char in s:
        if char == '(':
            open_count += 1
        elif char == ')':
            close_count += 1
            
        # If there are more closing parentheses than opening ones, 
        # it means that the string is not balanced at this point
        if close_count > open_count:
            return "No"
    
    # After going through all characters in the string, check if the counts of 
    # opening and closing parentheses are equal. If they are, then the string is good.
    if open_count == close_count:
        return "Yes"
    else:
        return "No"
