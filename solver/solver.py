from ortools.sat.python import cp_model
from math import prod
from itertools import product

def combos(n, op, target, k):
    out = []
    for v in product(range(1, n + 1), repeat=k):
        if k == 1: ok = v[0] == target
        elif op == '+': ok = sum(v) == target
        elif op == '*': ok = prod(v) == target
        elif op == '-': ok = k == 2 and abs(v[0] - v[1]) == target
        elif op == '/': ok = k == 2 and (max(v) == target * min(v))
        else: ok = False
        if ok:
            out.append(v)
    return out

def solve(n, cages):
    m = cp_model.CpModel()
    x = {(r, c): m.NewIntVar(1, n, f'x_{r}_{c}') for r in range(n) for c in range(n)}
    for i in range(n):
        m.AddAllDifferent([x[(i,c)] for c in range(n)])
        m.AddAllDifferent([x[(r,i)] for r in range(n)])

    for target, op, cells  in cages:
        m.AddAllowedAssignments([x[c] for c in cells], combos(n, op, target, len(cells)))

    solver = cp_model.CpSolver()
    status = solver.Solve(m)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return [[solver.Value(x[(r, c)]) for c in range(n)] for r in range(n)]