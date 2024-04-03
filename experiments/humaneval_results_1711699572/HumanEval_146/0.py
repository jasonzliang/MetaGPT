
def specialFilter(nums):
    def isOddDigit(n):
        return int(str(n)[0]) % 2 != 0 and int(str(n)[-1]) % 2 != 0

    count = 0
    for num in nums:
        if num > 10 and isOddDigit(num):
            count += 1
    return count
