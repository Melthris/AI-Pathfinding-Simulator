# ai_pathfinding_sim.py
# Reading references that assisted me with this:
# https://www.codinglad.com/blogs/ai-search-algorithms-comparison
# https://www.designgurus.io/answers/detail/how-to-solve-tree-and-graph-problems-in-coding-interviews


from collections import deque
import heapq
import time
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Our Maze array is here
# 1 = Wall
# 0 = Void
# -1 = Player
#  9 = Goal
maze = [
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
[1,0,-1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,0,1],
[1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1],
[1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,9,1,0,1],
[1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,1,0,1],
[1,0,1,1,1,1,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1],
[1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1,0,1],
[1,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,1,0,0,0,1,0,1],
[1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1],
[1,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

rows, cols = len(maze), len(maze[0])

def find_playerstart():
    return next(((y, x) for y in range(rows) for x in range(cols) if maze[y][x] == -1), None)

def find_goal():
    return next(((y, x) for y in range(rows) for x in range(cols) if maze[y][x] == 9), None)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def print_path(path):
    temp = [row[:] for row in maze]
    for y, x in path:
        if temp[y][x] == 0:
            temp[y][x] = 2
    for row in temp:
        print(''.join([
            Fore.RED + 'P' if cell == -1 else
            Fore.BLUE + 'G' if cell == 9 else
            Fore.GREEN + '█' if cell == 1 else
            Fore.YELLOW + '*' if cell == 2 else
            ' ' for cell in row
        ]))
# These values are adding weighting to the Path, Steps and Nodes Explored
# I'm doing this because this is a step I missed initially which causes
# algorithms to not properly sort from best to worst based on efficiency/
#
# This will only be called when comparing all algorithms against one another
# and then ranking on efficiency as determined by these weightings on the metrics

# Breadth First-Search
def bfs():
    start, goal = find_playerstart(), find_goal()
    queue, visited = deque([(start, [start])]), set()
    steps = 0
    while queue:
        (y, x), path = queue.popleft(); steps += 1
        if (y, x) == goal: return path, steps, len(visited)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < rows and 0 <= nx < cols and
                maze[ny][nx] != 1 and (ny, nx) not in visited):
                visited.add((ny, nx))
                queue.append(((ny, nx), path + [(ny, nx)]))
    return None, steps, len(visited)

# Depth First Search
def dfs():
    start, goal = find_playerstart(), find_goal()
    stack, visited = [(start, [start])], set()
    steps = 0
    while stack:
        (y, x), path = stack.pop(); steps += 1
        if (y, x) == goal: return path, steps, len(visited)
        if (y, x) in visited: continue
        visited.add((y, x))
        for dy, dx in [(1,0), (-1,0), (0,1), (0,-1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and maze[ny][nx] != 1:
                stack.append(((ny, nx), path + [(ny, nx)]))
    return None, steps, len(visited)

# Greedy Best-First Search
def greedy():
    start, goal = find_playerstart(), find_goal()
    heap, visited = [(heuristic(start, goal), start, [start])], set()
    steps = 0
    while heap:
        _, current, path = heapq.heappop(heap); steps += 1
        if current == goal: return path, steps, len(visited)
        if current in visited: continue
        visited.add(current)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < rows and 0 <= nx < cols and maze[ny][nx] != 1:
                heapq.heappush(heap, (heuristic((ny, nx), goal), (ny, nx), path + [(ny, nx)]))
    return None, steps, len(visited)

# Dijkstra's Algorithm. I added this simply because I like the name and it's referenced in
# Artifical Intelligence: A Modern Approach
def dijkstra():
    start, goal = find_playerstart(), find_goal()
    heap, visited = [(0, start, [start])], set()
    steps = 0
    while heap:
        cost, current, path = heapq.heappop(heap); steps += 1
        if current == goal: return path, steps, len(visited)
        if current in visited: continue
        visited.add(current)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < rows and 0 <= nx < cols and maze[ny][nx] != 1:
                heapq.heappush(heap, (cost + 1, (ny, nx), path + [(ny, nx)]))
    return None, steps, len(visited)

# A* Search
def astar():
    start, goal = find_playerstart(), find_goal()
    heap, visited = [(heuristic(start, goal), 0, start, [start])], set()
    while heap:
        est, cost, current, path = heapq.heappop(heap)
        if current == goal: return path, cost, len(visited)
        if current in visited: continue
        visited.add(current)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < rows and 0 <= nx < cols and maze[ny][nx] != 1:
                heapq.heappush(heap, (cost + 1 + heuristic((ny, nx), goal), cost + 1, (ny, nx), path + [(ny, nx)]))
    return None, 0, len(visited)

# Iterative Deepening Depth-First Search
def iddfs():
    goal = find_goal()
    for depth in range(1, 1000):
        visited = set()
        result = dls(find_playerstart(), goal, depth, [find_playerstart()], visited, 0)
        if result is not None:
            return result[0], result[1], len(visited)
    return None, 0, 0

def dls(current, goal, limit, path, visited, steps):
    if current == goal:
        return path, steps
    if limit <= 0:
        return None
    visited.add(current)
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = current[0] + dy, current[1] + dx
        if 0 <= ny < rows and 0 <= nx < cols and maze[ny][nx] != 1 and (ny, nx) not in visited:
            result = dls((ny, nx), goal, limit - 1, path + [(ny, nx)], visited, steps + 1)
            if result is not None:
                return result
    return None

# This runs all of our algoritms defined above
# and will later sort from best to worst performing algorithm
def run_all():
    algorithms = {
        "BFS": bfs,
        "DFS": dfs,
        "Greedy Best-First": greedy,
        "Dijkstra": dijkstra,
        "A*": astar,
        "IDDFS": iddfs
    }
    results = []
    for name, func in algorithms.items():
        path, steps, explored = func()
        if path:
            results.append((name, len(path), steps, explored))

    # Normalize metrics
    max_path = max(r[1] for r in results)
    max_steps = max(r[2] for r in results)
    max_explored = max(r[3] for r in results)

    scored_results = []
    for name, path_len, steps, explored in results:
        norm_path = path_len / max_path if max_path else 0
        norm_steps = steps / max_steps if max_steps else 0
        norm_explored = explored / max_explored if max_explored else 0

        # I didn't originally have weighting but found quickly that not opting for weighting
        # with these metrics didn't really give an honest "most optimal algorithm" result.
        # By adding weight and normalising that to the metrics below, we can gauge
        # which algorithm is the most efficient at getting the player to the goal state.
        score = (0.5 * norm_path) + (0.2 * norm_steps) + (0.3 * norm_explored)
        scored_results.append((name, path_len, steps, explored, score))

    scored_results.sort(key=lambda x: x[4])

    print(Fore.CYAN + "\n--- Ranked Results (Score (with weighting): lower is better) ---\n")
    for name, path_len, steps, explored, score in scored_results:
        print(f"{name}: Path={path_len}, Steps={steps}, Explored={explored}, Score={score:.3f}")

    best_name = scored_results[0][0]
    best_func = algorithms[best_name]
    best_path, _, _ = best_func()

    print(Fore.YELLOW + f"\nBest performing algorithm: {best_name}\n")
    print_path(best_path)
    

def menu():
    print(Fore.YELLOW + "=================================")
    print(Fore.YELLOW + "| AI Pathfinding Simulator v0.2 |")
    print(Fore.YELLOW + "=================================")
    print("")
    print("Player = " + Fore.RED + "P")
    print("Goal = " + Fore.BLUE + "G")
    print("Walls = " + Fore.GREEN + "█")
    print("Player path to Goal = " + Fore.YELLOW + "*")
    print("")
    print("Which algorithm would you like to test?:")
    print("1. BFS")
    print("2. DFS")
    print("3. Greedy Best-First")
    print("4. Dijkstra")
    print("5. A*")
    print("6. Iterative Deepening DFS")
    print("7. Compare All")
    return input("Select Choice: ")

if __name__ == '__main__':
    choice = menu()
    if choice == '1': name, func = "BFS", bfs
    elif choice == '2': name, func = "DFS", dfs
    elif choice == '3': name, func = "Greedy Best-First", greedy
    elif choice == '4': name, func = "Dijkstra", dijkstra
    elif choice == '5': name, func = "A*", astar
    elif choice == '6': name, func = "IDDFS", iddfs
    elif choice == '7': run_all(); exit()
    else: name, func = "BFS", bfs
    print(Fore.BLUE + f"\nRunning {name}...\n")
    start = time.time()
    path, steps, explored = func()
    if path:
        print(f"Found path. Length={len(path)}, Steps={steps}, Nodes Explored={explored}\n")
        print_path(path)
    else:
        print(Fore.RED + "Error: No path found.")
    print(f"Time taken: {time.time() - start:.4f}s")
