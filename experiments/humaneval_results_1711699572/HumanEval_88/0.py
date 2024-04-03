
def sort_array(arr):
    if not arr:
        return []
    elif len(arr) == 1:
        return [arr[0]]
    else:
        sum = arr[0] + arr[-1]
        if sum % 2 == 0:
            return sorted(arr, reverse=True)
        else:
            return sorted(arr)
