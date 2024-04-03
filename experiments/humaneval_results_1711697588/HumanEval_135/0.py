def can_arrange(arr):
    for i in range(len(arr) - 1, 0, -1):
        if arr[i] < arr[i-1]:
            return i
    return -1

print(can_arrange([1,2,4,3,5])) # Output: 3
print(can_arrange([1,2,3])) # Output: -1 