from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import functools
import sys

import requests
from z3 import *

# (left tile value, right tile value)
Domino = tuple[int, int]

# (row, col)
Cell = tuple[int, int]

class Constraint(Enum):
    empty = "empty"
    sum = "sum"
    equals = "equals"
    unequal = "unequal"
    less = "less"
    greater = "greater"

@dataclass
class Region:
    cells: list[Cell]
    constraint: Constraint
    target: int | None

    def extents(self):
        return max(c[0] for c in self.cells), max(c[1] for c in self.cells)

@dataclass
class Puzzle:
    dominoes: list[Domino]
    regions: list[Region]

    def extents(self):
        return max(e.extents()[0] for e in self.regions), max(e.extents()[1] for e in self.regions)

    def size(self):
        m, n = self.extents()
        return (m+1, n+1)

def read_tuples(source: list) -> list[tuple]:
    return [tuple(d) for d in source]

def read_region(source: dict) -> list[Region]:
    cells = read_tuples(source["indices"])
    constr = Constraint[source["type"]]
    target = int(source["target"]) if "target" in source else None
    return Region(cells, constr, target)

def read_puzzle(source: dict) -> Puzzle:
    return Puzzle(
        dominoes = read_tuples(source["dominoes"]),
        regions = [read_region(r) for r in source["regions"]]
    )

def solve(puzzle: Puzzle):
    s = Solver()
    m, n = puzzle.size()

    def in_bounds(i, j):
        return 0 <= i < m and 0 <= j < n

    cell_tiles = [ [ Int(f"cd_{i}_{j}") for j in range(n) ] for i in range(m) ]
    cell_values = [ [ Int(f"cv_{i}_{j}") for j in range(n) ] for i in range(m) ]
    num_tiles = 2 * len(puzzle.dominoes)

    # maps a cell to its corresponding puzzle region
    cell_to_region = {}
    for region in puzzle.regions:
        for cell in region.cells:
            cell_to_region[cell] = region

    for i in range(m):
        for j in range(n):
            ct = cell_tiles[i][j]
            cv = cell_values[i][j]

            if (i,j) not in cell_to_region:
                # cell is out of bounds. nothing can be placed here.
                s.add(ct == -1)
                s.add(cv == -1)
                continue

            # let's consider assigning any tile.
            s.add(0 <= ct, ct < num_tiles)
            for tid in range(num_tiles):
                # if tid is assigned to this cell...
                is_assigned = ct == tid
                
                # make sure cell value matches tile value.
                val_matches = cv == puzzle.dominoes[tid // 2][tid % 2]

                # make sure exactly one neighbor is sibling tile
                sibling_tid = (tid // 2) * 2 + (1 - (tid % 2)) # computes sibling tile's id
                neighbors = [ (i, j+1), (i+1, j), (i, j-1), (i-1, j) ]
                in_bounds_neighbors = [ (k,l) for k,l in neighbors if in_bounds(k, l) ]
                neighbor_is_sibling = [ (cell_tiles[k][l] == sibling_tid, 1) for (k,l) in in_bounds_neighbors ]
                exactly_one_sibling_neighbor = PbEq(neighbor_is_sibling, 1)
                s.add(Implies(is_assigned, And(val_matches, exactly_one_sibling_neighbor)))

    # each tile must be used exactly once
    for tid in range(num_tiles):
        preds = []
        for i in range(m):
            for j in range(n):
                preds.append((cell_tiles[i][j] == tid, 1))
        s.add(PbEq(preds, 1))

    # each region can introduce some constraint
    for r, region in enumerate(puzzle.regions):
        if region.constraint == Constraint.empty:
            # empty constraint: just in_boundsate we have a domino here
            for i, j in region.cells:
                s.add(cell_tiles[i][j] != -1)
        elif region.constraint == Constraint.equals:
            # each cell must equal some value k
            k = Int(f"k_{r}")
            for i, j in region.cells:
                s.add(cell_values[i][j] == k)
        elif region.constraint == Constraint.unequal:
            s.add(Distinct([cell_values[i][j] for i,j in region.cells]))
        else:
            # sum of cells must satisfy constraint
            total = Sum(cell_values[i][j] for i,j in region.cells)
            target = region.target

            if region.constraint == Constraint.less:
                s.add(total < target)
            elif region.constraint == Constraint.sum:
                s.add(total == target)
            else:
                s.add(total > target)

    result = s.check()
    if str(result) != "sat":
        raise RuntimeError(f"Solver returned {result}!")

    model = s.model()
    for i in range(m):
        for j in range(n):
            v = model.evaluate(cell_values[i][j]).as_long()
            t = model.evaluate(cell_tiles[i][j]).as_long()

            # if the tile to my right belongs to a different domino, separate us with a vertical pipe
            # this makes the ASCII output slightly easier to read :)
            next_different = False
            if j < n - 1:
                t_next = model.evaluate(cell_tiles[i][j+1]).as_long()
                if (t // 2) != (t_next // 2):
                    next_different = True

            ch = v if v >= 0 else "-"
            print(ch, end="|" if next_different else " ")

        print()

def main():
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    difficulty = sys.argv[2] if len(sys.argv) > 2 else "hard"
    url = f"https://www.nytimes.com/svc/pips/v1/{date}.json"
    with requests.get(url) as resp:
        source = resp.json()
        puzzle = read_puzzle(source[difficulty])
        solve(puzzle)

if __name__ == "__main__":
    main()
