from enum import Enum
from dataclasses import dataclass
import functools

import requests
from z3 import *

Domino = tuple[int]

Cell = tuple[int]

class Constraint(Enum):
    empty = "empty"
    sum = "sum"
    equals = "equals"
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
    e_i, e_j = puzzle.extents()

    m, n = e_i + 1, e_j + 1
    def valid(i, j):
        return 0 <= i < m and 0 <= j < n

    # maps a cell to its corresponding puzzle region
    cell_to_region = {}
    for region in puzzle.regions:
        for cell in region.cells:
            cell_to_region[cell] = region

    # cd_i_j = id of tile located in i,j
    cell_tiles = [ [ Int(f"cd_{i}_{j}") for j in range(n) ] for i in range(m) ]
    # cv_i_j = value of tile located in i,j
    cell_values = [ [ Int(f"cv_{i}_{j}") for j in range(n) ] for i in range(m) ]

    num_dominoes = len(puzzle.dominoes)
    num_tiles = 2 * num_dominoes 

    for i in range(m):
        for j in range(n):
            ct = cell_tiles[i][j]
            cv = cell_values[i][j]

            # out of bounds; nothing can be placed here
            if (i,j) not in cell_to_region:
                s.add(ct == -1)
                s.add(cv == -1)
                continue

            # let's consider assigning any tile of any domino
            s.add(0 <= ct, ct < num_tiles)
            for tid in range(num_tiles):
                # if this tile is assigned to i,j....
                antecedent = ct == tid
                
                # make sure cells[i][j] = this tile's value
                val_matches = cv == puzzle.dominoes[tid // 2][tid % 2]

                # make sure exactly one (valid) neighbor is sibling tile
                sibling_tid = (tid // 2) * 2 + (1 - (tid % 2))
                neighbors = [ (i, j+1), (i+1, j), (i, j-1), (i-1, j) ]
                valid_neighbors = [ (k,l) for k,l in neighbors if valid(k, l) ]
                neighbor_is_sibling = [ cell_tiles[k][l] == sibling_tid for (k,l) in valid_neighbors ]
                exactly_one_sibling_neighbor = PbEq([(n, 1) for n in neighbor_is_sibling], 1)

                s.add(Implies(antecedent, And(val_matches, exactly_one_sibling_neighbor)))

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
            # empty constraint: just validate we have a domino here
            for i, j in region.cells:
                s.add(cell_tiles[i][j] != -1)
        elif region.constraint == Constraint.equals:
            # each cell must equal some value k
            k = Int(f"k_{r}")
            for i, j in region.cells:
                s.add(cell_values[i][j] == k)
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

    print(s.check())
    model = s.model()
    for i in range(m):
        for j in range(n):
            v = model.evaluate(cell_values[i][j]).as_long()
            t = model.evaluate(cell_tiles[i][j]).as_long()

            next_different = False
            if j < n - 1:
                t_next = model.evaluate(cell_tiles[i][j+1]).as_long()
                if (t // 2) != (t_next // 2):
                    next_different = True

            ch = v if v >= 0 else "-"
            print(ch, end="|" if next_different else " ")

        print()

def main():
    url = "https://www.nytimes.com/svc/pips/v1/2025-08-23.json"
    with requests.get(url) as resp:
        source = resp.json()
        puzzle = read_puzzle(source["hard"])
        solve(puzzle)

if __name__ == "__main__":
    main()
