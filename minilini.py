#!/usr/bin/env python3
"""
      ______________________________________________________________________

       ###  ### ###### ###  ## ######     ##     ###### ###  ## ######
      ########   ##   #### ##   ##       ##       ##   #### ##   ##
     ## ## ##   ##   ## ####   ##       ##       ##   ## ####   ##
    ##    ## ###### ##  ### ######     ###### ###### ##  ### ######
______________________________________________________________________

    mINIlINI - A very (very) small Linear Algebra library providing step-by-step solutions
                to linear algebra problems.

        Functions that include step-by-step solutions:
            ref         - Row Echelon Form
            rref        - Reduced Row Echelon Form
            det         - Determinant
            qr_dec      - QR Decomposition
            eig_qr      - Eigenvalues from QR
            det_qr      - Determinant from QR
            information - Give info about matrix
            pivot_pos   - Positions of the pivots

            TODO
            eigenvalues
            eigenvectors
            SVD
            diagonalize
"""

import time
import numpy as np
import nptyping as npt
import sympy

from fractions import Fraction
from typing import Tuple, Any

np.set_printoptions(precision=2)




class MatrixDimensionException(Exception):
    """
    Raise this exception when there are issues with the matrix dimensions.
    """
    def __init__(self,expression,message):
        self.expression = expression
        self.message = message




def pivot_positions(
    matrix: npt.NDArray[Any,Any]
) -> int:
    #if not rref_check(matrix):
    matrix = rref(matrix)
    m,n = matrix.shape
    positions = {key: [] for key in range(n)}
    for i in range(n):
        for j in range(m-1,-1,-1):
            if round(matrix[j][i],10) == 1:
                positions[j].append(i)
                break
    return [positions[key][0] for key in positions.keys() if positions[key] != []], matrix




def information(
    matrix: npt.NDArray[Any,Any],
    verbose: bool = False
) -> None:
    """
    Get relavent information about matrix 
        all from inv mat thm
    """
    size = matrix.shape
    matrix_ref, scaling_factors = ref(matrix)
    pivots, matrix_rref = pivot_positions(matrix_ref)
    free_vars = list(filter(lambda i: i not in pivots, list(range(size[1]))))
    rank = len(pivots)
    det = round(np.prod(scaling_factors),2)*(rank == size[1])
    nullity = size[1] - rank
    ld = "linearly INDEPENDENT." if rank == size[1] else "linearly DEPENDENT."
    # null = nullspace(matrix_rref,free_vars)
    if verbose:
        print("Information about the Matrix:\n")
        print(matrix)
        print("\nRow Echelon Form:\n")
        print(matrix_ref)
        print("\nReduced Row Echelon Form:\n")
        print(matrix_rref)
        print(f"\nRows: \t\t {size[0]}")
        print(f"Columns: \t {size[1]}")
        print(f"Rank: \t\t {rank}")
        print(f"Nullity: \t {nullity}")
        print(f"Invertible: \t {rank == size[1]}")
        if size[0] == size[1]:
            print(f"Determinant: \t {det}")
            print(f"Eigenvalues: \t {list(np.linalg.eig(matrix)[0])}")
        print(f"Basis of Col: \t {[list(matrix[:,i]) for i in pivots]}")
        # print(f"Basis of Nul: \t {null}")
        print(f"Free Vars: \t {[i+1 for i in free_vars]}")
        print(f"Columns of A are " + ld)




def ref(
    matrix: npt.NDArray[Any,Any],
    steps: bool = False,
    as_fractions: bool = False
) -> Tuple:
    """
    Perform Gaussian Elimination to obtain Row Echelon Form of Matrix in place

    Parameters:
        matrix - numpy array,
        steps - print steps of algorithm,
        as_fractions - type of data (fractions or floats)

    Examples:
        steps = True, as_fractions = True,
        A = np.array([[2,1,7,12,Fraction(39,2)],[7,16,Fraction(1,2),32,21],[3,3,2,5,6],[14,11,1,9,Fraction(2,3)]]),
        B = ref(A,steps=True,as_fractions=True)

        steps = False, as_fractions = False
        A = np.array([[2,1,7,12,19.5],[7,16,0.5,32,21],[3,3,2,5,6],[14,11,1,9,2/3]]),
        B = ref(A)
    """
    if not (isinstance(matrix.shape, tuple) and list(map(type, matrix.shape)) == [int, int]):
        raise MatrixDimensionException(f'Matrix shape = {matrix.shape}.',f'Matrix shape must be [int,int].')
    if as_fractions:
        matrix = matrix + Fraction()
    else:
        matrix = matrix.astype('float64')
    scaling_factors = []
    n,m = matrix.shape
    i,j = 0,0
    while i < n and j < m:
        # if entry is 0
        if matrix.item((i,j)) == 0:
            k = i
            # find first nonzero entry
            while k < n and matrix.item((k,j)) == 0:
                k += 1
            # swap these if k < n
            if k < n:
                matrix[[i,k]] = matrix[[k,i]]
                scaling_factors.append(-1)
                if steps:
                    print(f'Swap Row {i+1} and Row {k+1}.')
                    print(matrix)
                    print('\n')
            else:
                i,j = i+1,j+1
                continue
        # now the entry is not zero
        if i < n-1 and j < m:
            # scale the row so matrix(i,j) == 1
            normfactor1 = matrix.item((i,j))
            scaling_factors.append(normfactor1)
            for l in range(m):
                matrix.itemset((i,l),round((1/normfactor1)*matrix.item((i,l)),10))
            if steps and normfactor1 != 1:
                print(f'Scale Row {i+1} by {normfactor1}.')
                print(matrix)
                print('\n')
            # use row replacement so all entries in column j below row i have 0
            for x in range(i+1,n):
                normfactor2 = matrix[x][j]
                for y in range(j,m):
                    matrix.itemset((x,y),round(matrix.item((x,y)) - normfactor2*matrix.item((i,y)),10))
                if steps:
                    print(f'Subtract {normfactor2} * Row {i+1} from Row {x+1}.')
                    print(matrix)
                    print('\n')
        if i == n-1:
            normfactor3 = matrix.item((i,j))
            scaling_factors.append(normfactor3)
            for l in range(m):
                matrix.itemset((i,l),round((1/normfactor3)*matrix.item((i,l)),10))
            if steps:
                print(f'Scale Row {i+1} by {normfactor3}.')
                print(matrix)
                print('\n')
        i,j = i+1,j+1
    return matrix, scaling_factors




def ref_check(
    matrix: npt.NDArray[Any,Any]
) -> bool:
    """
    Determine whether matrix is in row echelon form
    """
    if not (isinstance(matrix.shape, tuple) and list(map(type, matrix.shape)) == [int, int]):
        raise MatrixDimensionException(f'Matrix shape = {matrix.shape}.',f'Matrix shape must be [int,int].')
    n, m = matrix.shape
    for i in range(min(n,m)):
        if matrix[i][i] not in [0,1]:
            return False
        else:
            for j in range(i+1,n):
                if matrix[j][i] != 0:
                    return False
    return True




def rref(
    matrix: npt.NDArray[Any,Any],
    steps: bool = False,
    as_fractions: bool = False
) -> npt.NDArray[Any,Any]:
    """
    Finish the Gaussian Elimination Algorithm to obtain rref

    Parameters:
        matrix - numpy array,
        steps - print steps of algorithm,
        as_fractions - type of data (fractions or floats)
    """
    if not (isinstance(matrix.shape, tuple) and list(map(type, matrix.shape)) == [int, int]):
        raise MatrixDimensionException(f'Matrix shape = {matrix.shape}.',f'Matrix shape must be [int,int].')
    matrix_ref, _ = ref(matrix)
    if as_fractions:
        matrix_ref = matrix_ref + Fraction()
    else:
        matrix_ref = matrix_ref.astype('float64')
    n, m = matrix.shape
    i = 1
    while i < min(n,m) and round(matrix_ref[i][i],10) in [0,1]:
        for j in range(i):
            normfactor = matrix_ref[j][i]
            for k in range(m):
                #print(f'A[{j},{k}] = {matrix_ref[j][k]} - {normfactor} * {matrix_ref[i][k]}')
                matrix_ref.itemset((j,k), round(matrix_ref[j][k] - normfactor*matrix_ref[i][k],10))
            if steps:
                print(f'Subtract {normfactor} * Row {i+1} from Row {j+1}')
                print(matrix_ref)
        i += 1
    return matrix_ref




def rref_check(
    matrix: npt.NDArray[Any,Any]
) -> bool:
    """
    Check whether matrix is in rref
    TODO: Not working - needs to be fixed
    """
    if not (isinstance(matrix.shape, tuple) and list(map(type, matrix.shape)) == [int, int]):
        raise MatrixDimensionException(f'Matrix shape = {matrix.shape}.',f'Matrix shape must be [int,int].')
    n,m = matrix.shape
    i,j = 0,0
    while i < n-1 and j < m-1:
        if matrix[i][j] == 1:
            for k in range(n):
                if k != i and matrix[k][j] != 0:
                    print(k,j)
                    return False
        else:
            for k in range(i+1,n):
                if matrix[k][j] != 0:
                    print(k,j)
                    return False
        if matrix[i+1][j+1] == 0:
            i,j = i,j+1
        else:
            i,j = i+1,j+1
    return True




def is_invertible(
    matrix: npt.NDArray[Any,Any]
) -> bool:
    """
    Check the row echelon form to determine invertibility
    """
    if not (isinstance(matrix.shape, tuple) and list(map(type, matrix.shape)) == [int, int] and matrix.shape[0] == matrix.shape[1]):
        raise MatrixDimensionException(f'Matrix shape = {matrix.shape}.',f'Matrix shape must be square.')
    n,m = matrix.shape
    matrix = matrix.astype('float64')
    matrix_ref = ref(matrix)
    if n == m:
        for i in range(n):
            if matrix_ref[i][i] != 1:
                return False
        return True
    return False




def minor(
    mat: npt.NDArray[Any,Any],
    i: int,
    j: int
) -> npt.NDArray[Any,Any]:
    """
    Compute minor of mat by removing row i and column j
    """
    return mat[np.array(list(range(i))+list(range(i+1,mat.shape[0])))[:,np.newaxis],
               np.array(list(range(j))+list(range(j+1,mat.shape[1])))]




def det_from_ref(
    matrix: npt.NDArray[Any,Any],
    steps: bool = False,
    as_fractions: bool = False
) -> float:
    _, scaling_factors = ref(matrix,steps=steps,as_fractions=as_fractions)
    return np.prod(scaling_factors)




def det(
    matrix: npt.NDArray[Any,Any],
    shape: int,
    datatype: type = float
) -> float:
    """
    Calculate the determinant of the given matrix
    """
    if not (isinstance(matrix.shape, tuple) and list(map(type, matrix.shape)) == [int, int] and matrix.shape[0] == matrix.shape[1]):
        raise MatrixDimensionException(f'Matrix shape = {matrix.shape}.',f'Matrix shape must be square.')
    if shape == 1:
        return matrix[0][0]
    else:
        row1 = matrix[0]
        s = 0
        for i in range(len(row1)):
            minori = minor(matrix,0,i)
            if i % 2 == 0:
                s += row1[i]*det(minori, shape-1, datatype=datatype)
            else:
                s -= row1[i]*det(minori, shape-1, datatype=datatype)
        return s




def det_steps(
    matrix: npt.NDArray[Any,Any],
    shape: int,
    datatype: type = float
) -> str:
    """
    Calculate the determinant of the given matrix
    """
    if not (isinstance(matrix.shape, tuple) and list(map(type, matrix.shape)) == [int, int] and matrix.shape[0] == matrix.shape[1]):
        raise MatrixDimensionException(f'Matrix shape = {matrix.shape}.',f'Matrix shape must be square.')
    if shape == 1:
        return f'{matrix[0][0]}'
    else:
        row1 = matrix[0]
        s = ''
        for i in range(len(row1)):
            minori = minor(matrix,0,i)
            if i % 2 == 0:
                s += ' + '
            else:
                s += ' - '
            s += f'({row1[i]} * {det(minori, shape-1, datatype=datatype)})'
        return s




def eigenvalues(
    matrix: npt.NDArray[Any,Any],
    shape: int,
    tol: float = 1e-3
) -> Tuple:
    return




def householder(
    matrix: npt.NDArray[Any,Any],
    steps: bool = False
) -> Tuple:
    """
    Compute the Householder transformation of the given vector
    """
    v = matrix / (matrix[0] + np.copysign(np.linalg.norm(matrix), matrix[0]))
    v[0] = 1
    t = 2 / (v.T @ v)
    return v, t




def qr_decomposition(
    matrix: npt.NDArray[Any,Any],
    steps: bool = False,
    tol: float = 1e-10
) -> Tuple:
    """
    Compute the QR Decomposition of the given matrix
    """
    m,n = matrix.shape 
    Q = np.identity(m)
    R = matrix.copy()
    if steps:
        print(f'Shape of input matrix (A) is: {matrix.shape}.')
        print(f'Set Q = I_{m}.')
        print(f'Set R = A.\n')
    for i in range(0,n):
        v, t = householder(R[i:,i, np.newaxis])
        H = np.identity(m)
        H[i:,i:] -= t * (v @ v.T)
        R = H@R
        Q = H@Q
        if steps:
            print(f'Perform a Householder Transformation on the last {m-i} entries of column {i+1} obtaining v and t.')
            print(f'Set H = I_{m}.')
            print(f'Set the submatrix of H with rows {i+1}...{m} and columns {i+1}...{n} to H[{i+1}:{m},{i+1}:{n}] = H[{i+1}:{m},{i+1}:{n}] - t * (v @ v.transpose).')
            print(f'Now set R = H@R and Q = H@Q.\n')
    Q, R = Q[:n].T, np.triu(R[:n])
    return Q,R




def ut_tol(
    matrix: npt.NDArray[Any,Any],
    tol: float = 1e-5
) -> bool:
    if not (isinstance(matrix.shape, tuple) and list(map(type, matrix.shape)) == [int, int] and matrix.shape[0] == matrix.shape[1]):
        raise MatrixDimensionException(f'Matrix shape = {matrix.shape}.',f'Matrix shape must be square.')
    m, _ = matrix.shape 
    for i in range(1,m):
        for j in range(i):
            if tol < np.sqrt(np.power(matrix[i][j],2)):
                return False
    return True



def eigenvalues_from_qr(
    matrix: npt.NDArray[Any,Any],
    steps: bool = False,
    tol: float = 1e-5,
    t_complex: bool = False
) -> Tuple:
    """
    Works if the matrix has real eigenvalues using the QR decomp
    """
    s = time.time()
    e = s
    while not ut_tol(matrix,tol=tol) and e-s < 10:
        Q,R = qr_decomposition(matrix)
        matrix = R@Q
        e = time.time()
    if e - s >= 100:
        return "[[timeout finding eigenvalues]]"
    return np.diagonal(matrix)




def det_from_qr(
        matrix: npt.NDArray[Any,Any]
) -> float:
    Q,R = qr_decomposition(matrix)
    return (-1 if np.linalg.det(Q) < 0 else 1)*np.prod(np.diagonal(R))




if __name__ == '__main__':
    A = np.random.randint(0,4,(4,5))
    information(A,verbose=True)
