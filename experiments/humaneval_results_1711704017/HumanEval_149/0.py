
def sorted_list_sum(lst):
    """Write a function that accepts a list of strings as a parameter,
    deletes the strings that have odd lengths from it,
    and returns the resulted list with a sorted order.
    The order of the list should be ascending by length of each word, 
    and you should return the list sorted by that rule.
    If two words have the same length, sort the list alphabetically.
    """
    
    # Remove strings with odd lengths
    lst = [s for s in lst if len(s) % 2 == 0]
    
    # Sort the list by length and then alphabetically
    lst.sort(key=lambda x: (len(x), x))
    
    return lst
