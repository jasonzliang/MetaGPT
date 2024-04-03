
def total_match(lst1, lst2):
    """
    This function accepts two lists of strings and returns the list that has 
    total number of chars in the all strings of the list less than the other list.
    If the two lists have the same number of chars, return the first list.
    """
    
    # Calculate the total number of characters in each string for both lists
    lst1_total = sum(len(s) for s in lst1)
    lst2_total = sum(len(s) for s in lst2)

    # Compare the totals and return the appropriate list
    if lst1_total < lst2_total:
        return lst1
    else:
        return lst2
