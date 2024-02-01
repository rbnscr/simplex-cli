import argparse
import json

import numpy as np

class SimplexSolver:
    def __init__(self, simplex_input: dict, max_iter) -> None:
        self.simplex_input: dict = simplex_input
        self.max_iter = max_iter
        self.calc_dims()
        self.generate_solve_table()
        self.generate_variables()

        self.solve()


    def calc_dims(self):
        """
        Calculates the dimension in i (rows) and j (columns)
        self.dims = (i, j)
        """
        num_i = len(self.simplex_input.keys())
        num_j = len(self.simplex_input["target"]["x"]) + 1
        self.dims = (num_i, num_j)
    
    def generate_solve_table(self):
        sim_tab = np.empty(shape = self.dims)
        iter_i = 0
        for key in self.simplex_input.keys():
            sim_tab[iter_i][:-1] = self.simplex_input[key]["x"]
            sim_tab[iter_i][-1] = self.simplex_input[key]["rs"]
            iter_i += 1
        self.sim_tab = sim_tab

    def generate_variables(self):
        self.var_i: list = list(self.simplex_input.keys())
        var_j: list = []
        for j in range(self.dims[1]-1):
            var_j.append(f"x{j+1}")
        var_j.append("rs")
        self.var_j = list(var_j)

    def solve(self):

        solved = False
        self.n_iter = 0
        self.print_result()
        while solved is False or self.n_iter >= self.max_iter:
            old_sim_tab = self.sim_tab
            piv_col = choose_pivot_column(old_sim_tab)
            piv_row = choose_pivot_row(old_sim_tab, piv_col)

            # Calc new pivot element
            new_sim_tab = np.copy(old_sim_tab)
            piv_elem_old = old_sim_tab[piv_row, piv_col] 
            piv_elem_new = 1 / piv_elem_old
            new_sim_tab[piv_row, piv_col] = piv_elem_new

            # Exchange x and y
            new_var_j = self.var_i[piv_row]
            new_var_i = self.var_j[piv_col]
            self.var_i[piv_row] = new_var_i
            self.var_j[piv_col] = new_var_j

            new_sim_tab = calc_pivot_row(new_sim_tab, old_sim_tab, piv_col, piv_row, piv_elem_old)
            new_sim_tab = calc_pivot_column(new_sim_tab, old_sim_tab, piv_col, piv_row, piv_elem_old)
            new_sim_tab = calc_elements(new_sim_tab, old_sim_tab, piv_col, piv_row)

            solved = check_solved(new_sim_tab)

            self.sim_tab = np.copy(new_sim_tab)
            self.n_iter += 1
            self.print_result()

        self.solved_sim_tab = new_sim_tab
        print(f"Done after {self.n_iter} iterations.")
    
    def print_result(self):
        output_str = f'i = {self.n_iter} \t'
        for element in self.var_j:
            output_str += f"{element}\t"
        output_str += "\n"
        row = 0
        for element in self.var_i:
            output_str += f"{element}\t"
            for num in self.sim_tab[row]:
                output_str += f"{num:.3f}\t"
            output_str += "\n"
            row += 1
        print(output_str)


def check_solved(sim_tab: np.ndarray) -> bool:
    return all(sim_tab[0] >= 0)

def calc_pivot_row(new_sim_tab: np.ndarray, old_sim_tab: np.ndarray, piv_col: int, piv_row: int, piv_elem: float):
    for j in range(len(new_sim_tab[0])):
        if j != piv_col:
            new_sim_tab[piv_row, j] = old_sim_tab[piv_row, j] / piv_elem
    return new_sim_tab

def calc_pivot_column(new_sim_tab: np.ndarray, old_sim_tab: np.ndarray, piv_col: int, piv_row: int, piv_elem: float):
    for i in range(len(new_sim_tab)):
        if i != piv_row:
            new_sim_tab[i, piv_col] = -old_sim_tab[i, piv_col] / piv_elem
    return new_sim_tab

def calc_elements(new_sim_tab: np.ndarray, old_sim_tab: np.ndarray, piv_col: int, piv_row: int):
    for i in range(len(new_sim_tab)):
        if i != piv_row:
            for j in range(len(new_sim_tab[0])):
                if j != piv_col:
                    new_sim_tab[i, j] = old_sim_tab[i, j] - old_sim_tab[i, piv_col] * new_sim_tab[piv_row, j]
    return new_sim_tab


def choose_pivot_column(sim_tab: np.ndarray) -> int:
    piv_col: int = np.argmin(sim_tab[0])
    print(f"Pivot column index: {piv_col}")
    return piv_col

def choose_pivot_row(sim_tab: np.ndarray, piv_col: int) -> int:
    full_col_s = sim_tab[1:, piv_col]
    full_col_rs = sim_tab[1:,-1]
    piv_row_all = np.divide(full_col_rs, full_col_s, out=np.zeros_like(full_col_rs), where=full_col_s!=0)
    piv_row_all[piv_row_all<=0] = np.inf
    piv_row = np.argmin(piv_row_all) + 1
    print(f"Pivot row index: {piv_row}")
    return piv_row

def main():
    parser = argparse.ArgumentParser(
        prog="Simplex Method with Text Output",
        description="Solves a linear programming problem using the simplex method. Output of the iteration steps is provided."
    )
    parser.add_argument('filename', help='path to a json-file')

    args = parser.parse_args()
    json_file = open(args.filename)
    test_json: dict = json.load(json_file)
    SimplexSolver(test_json["input"], test_json["max_iter"])

if __name__ == "__main__":
    main()