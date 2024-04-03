
def match_parens(lst):
    # Concatenate the two strings in some order.
    s = lst[0] + lst[1]
    balance = 0
    for c in s:
        if c == '(':
            balance += 1
        elif c == ')':
            balance -= 1
        # If at any point the balance goes negative, it means we have more closing brackets than opening ones.
        if balance < 0:
            return 'No'
    # After going through all characters in s, if the balance is not zero, it means there are unbalanced parentheses.
    if balance != 0:
        return 'No'
    return 'Yes'
