
def strange_sort_list(lst):
    '''
    Given list of integers, return list in strange order.
    Strange sorting, is when you start with the minimum value,
    then maximum of the remaining integers, then minimum and so on.

    Examples:
    strange_sort_list([1, 2, 3, 4]) == [1, 4, 2, 3]
    strange_sort_list([5, 5, 5, 5]) == [5, 5, 5, 5]
    strange_sort_list([]) == []
    '''
    
    # Error handling for invalid inputs
    if not all(isinstance(i, int) for i in lst):
        raise ValueError("All elements of the list must be integers.")
        
    sorted_lst = sorted(lst)  # Sorting the list in ascending order
    result = []
    
    while len(sorted_lst) > 0:
        # Adding minimum and maximum values to the result list
        result.append(sorted_lst.pop(0))  
        if sorted_lst:  # Checking if there are still elements in the list
            result.append(sorted_lst.pop())
            
    return result
