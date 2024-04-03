
def exchange(lst1, lst2):
    # Check if all elements in lst1 are even
    for num in lst1:
        if num % 2 != 0:
            return "NO"
            
    # Check if there is any odd number in lst2
    for num in lst2:
        if num % 2 != 0:
            return "YES"
    
    return "NO"
