
def minPath(grid, k):
    N = len(grid)
    dp = [[[float('inf')] * (k+1) for _ in range(N)] for _ in range(N)]
    path = [[[[None] * (k+1) for _ in range(N)] for _ in range(N)] for _ in range(N)]
    
    for x in range(N):
        for y in range(N):
            dp[x][y][0] = grid[x][y]
            path[x][y][0] = [grid[x][y]]
            
    for step in range(1, k+1):
        for x in range(N):
            for y in range(N):
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N:
                        if dp[nx][ny][step-1] < dp[x][y][step]:
                            dp[x][y][step] = dp[nx][ny][step-1]
                            path[x][y][step] = path[nx][ny][step-1] + [grid[x][y]]
                        elif dp[nx][ny][step-1] == dp[x][y][step]:
                            if path[nx][ny][step-1] + [grid[x][y]] < path[x][y][step]:
                                path[x][y][step] = path[nx][ny][step-1] + [grid[x][y]]
    
    min_path, min_val = None, float('inf')
    for x in range(N):
        for y in range(N):
            if dp[x][y][k-1] < min_val:
                min_val = dp[x][y][k-1]
                min_path = path[x][y][k-1]
    
    return min_path[:k]
