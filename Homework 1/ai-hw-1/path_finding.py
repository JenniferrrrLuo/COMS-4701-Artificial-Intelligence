import numpy as numpy
from queue import PriorityQueue
from utils.utils import PathPlanMode, Heuristic, cost, expand, visualize_expanded, visualize_path
import numpy as np


def uninformed_search(grid, start, goal, mode: PathPlanMode):
    """ Find a path from start to goal in the gridworld using 
    BFS or DFS.
    
    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        mode (PathPlanMode): The search strategy to use. Must
        specify either PathPlanMode.DFS or PathPlanMode.BFS.
    
    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
    """

    frontier = [start]
    frontier_sizes = []
    expanded = []
    reached = {start: None}

    # TODO:
    
    # BFS: First In First Out
    # DFS: Last In First Out
    
    while len(frontier) != 0:
        
        frontier_sizes.append(len(frontier))
        
        if mode == PathPlanMode.DFS:
            cell = frontier.pop()
        elif mode == PathPlanMode.BFS: 
            cell = frontier.pop(0)
        # else: 
        #     ValueError
        
        if cell == goal: 
            current = goal
            path = [current]
            while reached[current] is not None:
                path.append(reached[current])
                current = reached[current]
            path.reverse()
            return path, expanded, frontier_sizes
        
        expanded.append(cell)
        
        for child in expand(grid, cell):
            if child not in reached: 
                reached[child] = cell; 
                frontier.append(child)
        
    path = []
    return path, expanded, frontier_sizes



def a_star(grid, start, goal, mode: PathPlanMode, heuristic: Heuristic, width):
    """ Performs A* search or beam search to find the
    shortest path from start to goal in the gridworld.
    
    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        mode (PathPlanMode): The search strategy to use. Must
        specify either PathPlanMode.A_STAR or
        PathPlanMode.BEAM_SEARCH.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.
        width (int): The width of the beam search. This should
        only be used if mode is PathPlanMode.BEAM_SEARCH.
    
    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
    """

    frontier = PriorityQueue()
    frontier.put((0, start))
    frontier_sizes = []
    expanded = []
    reached = {start: {"cost": cost(grid, start), "parent": None}}

    # TODO:
        
    while frontier.qsize() != 0: 
        
        frontier_sizes.append(frontier.qsize())
        
        cell_priority, cell = frontier.get()
        
        if cell == goal: 
            current = goal
            path = [current]
            while reached[current]["parent"] is not None: 
                path.append(reached[current]["parent"])
                current = reached[current]["parent"]
            path.reverse()    
            return path, expanded, frontier_sizes
        
        expanded.append(cell)
        
        for child in expand(grid, cell):
            
            if heuristic == Heuristic.MANHATTAN:
                h = abs(child[0] - goal[0]) + abs(child[1] - goal[1])
            elif heuristic == Heuristic.EUCLIDEAN:
                h = np.sqrt((child[0] - goal[0]) ** 2 + (child[1] - goal[1]) ** 0.5)
            
            # backward cost
            g = reached[cell]["cost"] + cost(grid, child)
            
            priority = g + h
            
            if child not in reached or g < reached[child]["cost"]: 
                reached[child] = {"cost": g, "parent": cell} 
                frontier.put((priority, child))
                
        if mode == PathPlanMode.BEAM_SEARCH: 
            new_frontier = PriorityQueue()
            for i in range(min(width, frontier.qsize())): 
                new_frontier.put(frontier.get())
            frontier = new_frontier  
        
    path = []
    return path, expanded, frontier_sizes


def ida_star(grid, start, goal, heuristic: Heuristic):
    """ Performs IDA* search to find the shortest path from
    start to goal in the gridworld.

    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.
        
    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
    """

    bound = 0
    frontier_sizes = []
    while True:
        path, expanded, frontier_size, new_bound = __dfs_ida_star(grid, start, goal, heuristic, bound)
        frontier_sizes += frontier_size
        
        if len(path) > 0 or np.isinf(new_bound):
            return path, expanded, frontier_sizes
        else:
            bound = new_bound


def __dfs_ida_star(grid, start, goal, heuristic: Heuristic, bound):
    """ Helper function for IDA* search to find the shortest path
    from start to goal in the gridworld.

    Args:
        grid (numpy): NxN numpy array representing the world,
        with terrain features encoded as integers.
        start (tuple): The starting cell of the path.
        goal (tuple): The ending cell of the path.
        heuristic (Heuristic): The heuristic to use. Must
        specify either Heuristic.MANHATTAN or Heuristic.EUCLIDEAN.
        bound (float): Maximum allowable cost of expanded nodes.

    Returns:
        path (list): A list of cells from start to goal.
        expanded (list): A list of expanded cells.
        frontier_size (list): A list of integers containing
        the size of the frontier at each iteration.
        next_bound (float): New value of cost upper bound in
        next iteration of IDA*.
    """

    frontier = [start]
    frontier_sizes = []
    expanded = []
    reached = {start: {"cost": cost(grid, start), "parent": None}}
    next_bound = np.inf

    # TODO:
    
    while len(frontier) != 0: 
        
        frontier_sizes.append(len(frontier))
        
        cell = frontier.pop()
        
        if cell == goal: 
            current = goal
            path = [current]
            while reached[current]["parent"] is not None: 
                path.append(reached[current]["parent"])
                current = reached[current]["parent"]
            path.reverse()    
            return path, expanded, frontier_sizes, next_bound
        
        expanded.append(cell)
        
        for child in expand(grid, cell):
            
            if heuristic == Heuristic.MANHATTAN:
                h = abs(child[0] - goal[0]) + abs(child[1] - goal[1])
            elif heuristic == Heuristic.EUCLIDEAN:
                h = np.sqrt((child[0] - goal[0]) ** 2 + (child[1] - goal[1]) ** 2)
            
            # backward cost
            g = reached[cell]["cost"] + cost(grid, child)
            
            priority = g + h
            
            if child not in reached or g < reached[child]["cost"]: 
                if priority <= bound:
                    reached[child] = {"cost": g, "parent": cell}
                    frontier.append(child)
                else: 
                    next_bound = min(next_bound, priority)

    path = []
    return path, expanded, frontier_sizes, next_bound


def test_world(world_id, start, goal, h, width, animate, world_dir):
    print(f"Testing world {world_id}")
    grid = np.load(f"{world_dir}/world_{world_id}.npy")

    if h == 1 or h == 2:
        modes = [
            PathPlanMode.A_STAR,
            PathPlanMode.BEAM_SEARCH
        ]
    elif h == 3 or h == 4:
        h -= 2
        modes = [
            PathPlanMode.IDA_STAR
        ]
    else:
        modes = [
            PathPlanMode.DFS,
            PathPlanMode.BFS
        ]

    for mode in modes:

        search_type, path, expanded, frontier_size = None, None, None, None
        if mode == PathPlanMode.DFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "DFS"
        elif mode == PathPlanMode.BFS:
            path, expanded, frontier_size = uninformed_search(grid, start, goal, mode)
            search_type = "BFS"
        elif mode == PathPlanMode.A_STAR:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, 0)
            search_type = "A_STAR"
        elif mode == PathPlanMode.BEAM_SEARCH:
            path, expanded, frontier_size = a_star(grid, start, goal, mode, h, width)
            search_type = "BEAM_A_STAR"
        elif mode == PathPlanMode.IDA_STAR:
            path, expanded, frontier_size = ida_star(grid, start, goal, h)
            search_type = "IDA_STAR"
        
        if search_type != None:
            path_cost = 0
            for c in path:
                path_cost += cost(grid, c)

            print(f"Mode: {search_type}")
            print(f"Path length: {len(path)}")
            print(f"Path cost: {path_cost}")
            print(f"Number of expanded states: {len(frontier_size)}")
            print(f"Max frontier size: {max(frontier_size) if len(frontier_size) > 0 else 0}\n")
            if animate == 0 or animate == 1:
                visualize_expanded(grid, start, goal, expanded, path, animation=animate)
            else:
                visualize_path(grid, start, goal, path)