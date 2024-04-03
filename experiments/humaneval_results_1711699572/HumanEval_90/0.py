
def next_smallest(lst):
    if len(set(lst)) <= 1:  # If all elements are the same or list is empty
        return None
    else:
        lst = sorted(list(set(lst)))  # Remove duplicates and sort the list
        return lst[1]  # Return the second smallest element
