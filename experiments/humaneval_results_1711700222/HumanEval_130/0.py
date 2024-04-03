
def tribonacci(n):
    if n == 0: return [1]
    elif n == 1: return [1, 3]
    else:
        seq = [1, 3]
        for i in range(2, n+1):
            if i % 2 == 0:
                seq.append(int(1 + i / 2))
            else:
                seq.append(seq[i-1] + seq[i-2] + seq[i])
        return seq[:n+1]
