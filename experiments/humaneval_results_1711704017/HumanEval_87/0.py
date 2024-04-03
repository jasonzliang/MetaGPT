
def get_row(lst, x):
    """
    Find integers x in the list and return a list of tuples (row, columns), 
    starting with 0. Sort coordinates initially by rows in ascending order, 
    and sort coordinates of the row by columns in descending order.
    """
    result = []
    
    for i in range(len(lst)):
        for j in reversed(range(len(lst[i]))):
            if lst[i][j] == x:
                result.append((i, j))
                
    return sorted(result)
