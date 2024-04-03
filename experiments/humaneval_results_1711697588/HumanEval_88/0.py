
def sort_array(arr):
    if not arr:
        return []
    new_arr = arr[:]
    if (new_arr[0] + new_arr[-1]) % 2 == 0:
        new_arr.sort(reverse=True)
    else:
        new_arr.sort()
    return new_arr
