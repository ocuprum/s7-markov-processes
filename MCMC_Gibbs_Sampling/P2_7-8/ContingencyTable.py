import numpy as np
from Graph import Graph 

rng = np.random.default_rng()

class ContingencyTable:
    def __init__(self, table, r, c) -> None:
        self.table = table
        self.r = r
        self.c = c
        self.len_r = len(self.r)
        self.len_c = len(self.c)
        self.table_graph = Graph([tbl_to_tuple(self.table)], {tbl_to_tuple(self.table): 0}, {})
    
    def __is_chess(self, tbl):
        if tbl == [[1, 0], 
                   [0, 1]]:
            return True, 1
        elif tbl == [[0, 1], 
                     [1, 0]]:
            return True, 2
        return False
    
    def __random_square(self):
        r1, r2 = rng.integers(0, self.len_r), rng.integers(0, self.len_r)
        c1, c2 = rng.integers(0, self.len_c), rng.integers(0, self.len_c)

        while r1 >= r2 or c1 >= c2:
            if r1 >= r2:
                r1 = rng.integers(0, self.len_r)
                r2 = rng.integers(0, self.len_r)
            if c1 >= c2:
                c1 = rng.integers(0, self.len_c)
                c2 = rng.integers(0, self.len_c)

        tbl = [[0, 0], [0, 0]]
        tbl[0][0] = self.table[r1][c1]
        tbl[0][1] = self.table[r1][c2]
        tbl[1][0] = self.table[r2][c1]
        tbl[1][1] = self.table[r2][c2]

        return tbl, r1, r2, c1, c2

    def __change_chess(self, res, r1, r2, c1, c2):
        table = self.table.copy()
        if res == 1:
            table[r1][c1], table[r1][c2] = 0, 1
            table[r2][c1], table[r2][c2] = 1, 0
        elif res == 2:
            table[r1][c1], table[r1][c2] = 1, 0
            table[r2][c1], table[r2][c2] = 0, 1
        
        return table

    def next_not_MCMC(self):
        tbl, r1, r2, c1, c2 = self.__random_square()

        result = self.__is_chess(tbl)
        if type(result) is tuple:
            self.table = self.__change_chess(result[1], r1, r2, c1, c2)

        return self.table
    
    def next_MCMC(self):
        while True:
            tbl, r1, r2, c1, c2 = self.__random_square()
            result = self.__is_chess(tbl)
            if type(result) is tuple and result[0]:
                break
        
        tpl = tbl_to_tuple(self.table)

        prop = self.__change_chess(result[1], r1, r2, c1, c2)
        tpl_prop = tbl_to_tuple(prop)
        self.table_graph.V.append(tbl_to_tuple(prop))
        self.table_graph.v_degrees[tpl] = self.table_graph.v_degrees.get(tpl, 0) + 1
        self.table_graph.v_degrees[tpl_prop] = self.table_graph.v_degrees.get(tpl_prop, 0) + 1
        self.table_graph.E[tpl] = self.table_graph.E.get(tpl, []) + [tpl_prop]
        self.table_graph.E[tpl_prop] = self.table_graph.E.get(tpl_prop, []) + [tpl]

        d1, d2 = self.table_graph.v_degrees[tpl], self.table_graph.v_degrees[tpl_prop]
        alpha = min([1, d1 / d2])

        self.table = proposal(alpha, self.table, prop)

        return self.table

def tbl_to_tuple(tbl):
    lst = []

    for row in tbl:
        for el in row:
            lst.append(el)
    
    return tuple(lst)

def proposal(alpha, i, j):
    u = rng.uniform(0, 1)
    
    if u <= alpha: return j
    elif u > alpha: return i