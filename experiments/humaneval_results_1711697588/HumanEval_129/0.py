
def minPath(grid, k):
    N = len(grid)
    dp = [[[float('inf') for _ in range(k+1)] for _ in range(N*N+1)] for _ in range(N*N+1)]
    path = [[(0, 0) for _ in range(N*N+1)] for _ in range(N*N+1)]
    
    def get_neighbors(i, j):
        neighbors = []
        if i > 0:
            neighbors.append((i-1, j))
        if i < N - 1:
            neighbors.append((i+1, j))
        if j > 0:
            neighbors.append((i, j-1))
        if j < N - 1:
            neighbors.append((i, j+1))
        return neighbors
    
    for i in range(N):
        for j in range(N):
            dp[grid[i][j]][1] = 0
            path[grid[i][j]][1] = (i, j)
            
    for l in range(2, k+1):
        for i in range(1, N*N+1):
            for j in range(1, N*N+1):
                if dp[i][l] > dp[j][l-1] + abs(grid[path[j][l-1][0]][path[j][l-1][1]] - i):
                    dp[i][l] = dp[j][l-1] + abs(grid[path[j][l-1][0]][path[j][l-1][1]] - i)
                    path[i][l] = (path[j][l-1][0], path[j][l-1][1])
                for x, y in get_neighbors(path[j][l-1][0], path[j][l-1][1]):
                    if dp[i][l] > dp[grid[x][y]][l-1] + abs(grid[path[j][l-1][0]][path[j][l-1][1]] - i) + abs(grid[x][y] - grid[path[j][l-1][0]][path[j][l-1][1]]):
                        dp[i][l] = dp[grid[x][y]][l-1] + abs(grid[path[j][l-1][0]][path[j][l-1][1]] - i) + abs(grid[x][y] - grid[path[j][l-1][0]][path[j][l-1][1]])
                        path[i][l] = (x, y)
    
    min_val = float('inf')
    for i in range(1, N*N+1):
        if dp[i][k] < min_val:
            min_val = dp[i][k]
            start = i
            
    result = [grid[path[start][k][0]][path[start][k][1]]]
    for l in range(k, 1, -1):
        result.append(grid[path[result[-1]][l-1][0]][path[result[-1]][l-1][1]])
    
    return result[::-1]
