
def tribonacci(n):
    if n == 0: return []
    elif n == 1: return [3]
    elif n == 2: return [3, 1]
    else:
        seq = [3, 1, 2]
        for i in range(3, n+1):
            if i % 2 == 0:
                seq.append(seq[i-1] + seq[i-2] + seq[i-3])
            else:
                seq.append(1 + i // 2)
        return seq
    