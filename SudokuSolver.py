import numpy as np
from numba import njit

easy_sudoku = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 0],
                         [6, 0, 0, 1, 9, 5, 0, 0, 0],
                         [0, 9, 8, 0, 0, 0, 0, 6, 0],
                         [8, 0, 0, 0, 6, 0, 0, 0, 3],
                         [4, 0, 0, 8, 0, 3, 0, 0, 1],
                         [7, 0, 0, 0, 2, 0, 0, 0, 6],
                         [0, 6, 0, 0, 0, 0, 2, 8, 0],
                         [0, 0, 0, 4, 1, 9, 0, 0, 5],
                         [0, 0, 0, 0, 8, 0, 0, 7, 9]])

hard_sudoku = np.array([[2, 0, 0, 0, 8, 0, 0, 6, 0],
						   [6, 0, 0, 0, 0, 3, 1, 7, 0],
                           [0, 1, 0, 9, 0, 0, 0, 4, 0],
                           [0, 8, 4, 0, 1, 9, 0, 0, 0],
                           [9, 6, 0, 0, 0, 0, 0, 0, 0],
                           [0, 2, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 7, 0, 2, 9, 0],
                           [0, 0, 0, 0, 0, 5, 0, 0, 8],
                           [7, 0, 6, 0, 0, 0, 0, 0, 3]])

@njit
def good_array(array, n):
    ssum = np.sum(array)
    for j in range(1, n+1):
        ssum2 = np.sum(np.where(array==j, array+1, array))
        if (ssum2-ssum) > 1:
            return False
    return True

@njit
def v_constraint(array, n):
    _array = np.zeros(n)
    for h in range(n):
        for j in range(n):
            _array[j] = array[j*n+h]
        if not good_array(_array, n):
            return False
    return True

@njit
def h_constraint(array, n):
    _array = np.zeros(n)
    for h in range(n):
        for j in range(n):
            _array[j] = array[j+h*n]
        if not good_array(_array, n):
            return False
    return True

@njit
def squ_constraint(array, n):
    _array = np.zeros(n)
    nn = int(n**0.5)
    
    loo = np.sort(np.array([j+n*i for j in range(nn) for i in range(nn)]))
    for j in loo:
        p=0
        for c in range(nn):
            for r in range(nn):
                pos = nn*(j+nn*c)+r
                _array[p] = array[pos:pos+1][0]
                p+=1

        if not good_array(_array, n):
            return False
    return True

@njit
def _all_constraint(array, n):
    return v_constraint(array, n) &\
            h_constraint(array, n) &\
            squ_constraint(array, n)

def _copy(sdk, N):
    _array = np.zeros(N)
    for j in range(N):
        _array[j] = sdk[j]
    return _array

def replace(_array, idxx, value):
    value = [value] if type(value) in [np.int32, int] else value
    for idx, val in zip(idxx, value):
        _array[idx] = val
    return _array

def _get_comb_per_zero(sdk, n, N, idx):
    assert sdk[idx]==0
    _array = _copy(sdk, N)
    return np.array([j for j in range(1, n+1) \
                     if _all_constraint(replace(_array, idx, j), n)])

def find_zeros(sdk):
    _list = list()
    for j, numb in enumerate(sdk):
        if numb==0:
            _list.append([j])
    return np.array(_list)


def _get_comb_all_zeros(sdk, n, N):
    zeros_loc = find_zeros(sdk)
    _list = list()
    for j in zeros_loc:
        _list.append(_get_comb_per_zero(sdk, n, N, j))
    return zeros_loc, _list

def _valid_branch(_array, trayectory, n, N):
    array= _copy(_array, N)
    zeros = find_zeros(array)
    return _all_constraint(replace(array, zeros, trayectory), n)

def _new_trayectories(_sdk, current_t, nodes, j, J, n, N):
    sdk = _copy(_sdk, N)
    if j>=0:
        new_t = list()
        for tray in current_t:
            for val in nodes[J-j]:
                if _valid_branch(sdk, tray+[val], n, N):
                    new_t.append(tray+[val])

        return _new_trayectories(sdk, new_t, nodes, j-1, J, n, N)
    else:
        return current_t

class Sudoku():
    def __init__(self, st_sdk):

        self.st_sdk = st_sdk
        self.sdk_1d = st_sdk.flatten().astype(int)
        self.N = int(len(self.sdk_1d))
        self.n = int(self.N ** 0.5)
        assert self.all_constraint(self.sdk_1d)

    def all_constraint(self, sdk):
        return _all_constraint(sdk, self.n)

    def copy(self, sdk):
        return _copy(sdk, self.N)

    def get_comb_per_zero(self, sdk, idx):
        return _get_comb_per_zero(sdk, self.n, self.N, idx)

    def get_comb_all_zeros(self, sdk):
        return _get_comb_all_zeros(sdk, self.n, self.N)

    def is_trivial(self, sdk):
        zeros, combs = self.get_comb_all_zeros(sdk)
        _dummy = [True for j in combs if len(j)==1]
        return True if len(_dummy)>0 else False

    def solve_trivial(self, __sdk):
        def _solve_trivial(sdk):

            _sdk = self.copy(sdk)
            while self.is_trivial(_sdk):
                zeros, combs = self.get_comb_all_zeros(_sdk)
                for z, c in zip(zeros, combs):
                    if len(c)==1:
                        _sdk[z] = c[0]
            return _sdk
        sdk = _solve_trivial(__sdk)
        return sdk

    def new_trayectories(self, sdk):
        nodes = _get_comb_all_zeros(sdk, self.n, self.N)
        if len(nodes[0])==0:
            return nodes[0], np.array(list())
        l_nodes = len(nodes[1])
        j, J = [l_nodes-2] * 2
        ini_node = [[nodes[1][0][j]] for j in range(len(nodes[1][0]))]
        return nodes[0], _new_trayectories(sdk, ini_node,
                                           nodes[1][1:], j, J, self.n, self.N)[0]

    def solve_backt(self):
        _sdk = self.solve_trivial(self.sdk_1d)
        idxx, trayec = self.new_trayectories(_sdk)
        sdk = replace(_sdk, idxx, trayec)
        return np.array([sdk[j:j+self.n] for j in range(self.n)])

if __name__ == '__main__':
    sudoku = Sudoku(st_sdk=hard_sudoku)
    solved_sukdoku = sudoku.solve_backt()
    