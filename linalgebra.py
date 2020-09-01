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
            ref     - Row Echelon Form
            rref    - Reduced Row Echelon Form
            det     - Determinant

            TODO
            eigenvalues
            eigenvectors
            QR decomp
            SVD
            diagonalize
"""

import time
import numpy as np
import nptyping as npt

from fractions import Fraction
from typing import Tuple, Any




class MatrixDimensionException(Exception):
    """
    Raise this exception when there are issues with the matrix dimensions.
    """
    def __init__(self,expression,message):
        self.expression = expression
        self.message = message




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
        B = row_echelon_form(A,steps=True,as_fractions=True)

        steps = False, as_fractions = False
        A = np.array([[2,1,7,12,19.5],[7,16,0.5,32,21],[3,3,2,5,6],[14,11,1,9,2/3]]),
        B = row_echelon_form(A)
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
                matrix.itemset((i,l),(1/normfactor1)*matrix.item((i,l)))
            if steps and normfactor1 != 1:
                print(f'Scale Row {i+1} by {normfactor1}.')
                print(matrix)
                print('\n')
            # use row replacement so all entries in column j below row i have 0
            for x in range(i+1,n):
                normfactor2 = matrix[x][j]
                for y in range(j,m):
                    matrix.itemset((x,y),matrix.item((x,y)) - normfactor2*matrix.item((i,y)))
                if steps:
                    print(f'Subtract {normfactor2} * Row {i+1} from Row {x+1}.')
                    print(matrix)
                    print('\n')
        if i == n-1:
            normfactor3 = matrix.item((i,j))
            scaling_factors.append(normfactor3)
            for l in range(m):
                matrix.itemset((i,l),(1/normfactor3)*matrix.item((i,l)))
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
    while i < min(n,m) and matrix_ref[i][i] in [0,1]:
        if matrix_ref[i][i] == 1:
            for j in range(i):
                normfactor = matrix_ref[j][i]
                for k in range(m):
                    #print(f'A[{j},{k}] = {matrix_ref[j][k]} - {normfactor} * {matrix_ref[i][k]}')
                    matrix_ref.itemset((j,k), matrix_ref[j][k] - normfactor*matrix_ref[i][k])
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
    matrix_ref, scaling_factors = ref(matrix,steps=steps,as_fractions=as_fractions)
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
            s += f'({row1[i]} * {det(minori, shape-1, datatype=datatype, steps=True)})'
        return s




def eigenvalues(
    matrix: npt.NDArray[Any,Any],
    shape: int,
    datatype: type = float
) -> Tuple:
    return




if __name__ == '__main__':

    # A = np.array([[2,1,7,12,39/2],[7,16,1/2,32,21],[3,3,2,5,6],[14,11,1,9,2/3]])
    # print(f'REF A: {ref_check(A)}')
    # print(f'RREF A: {rref_check(A)}')
    # B = ref(A)
    # print(B)
    # print(f'REF B: {ref_check(B)}')
    # print(f'RREF B: {rref_check(B)}')
    # print(is_invertible(B))
    # print(rref(B,steps=True))
    # print(f'RREF B: {rref_check(B)}')

    C = np.array([[2,2],[2,3]])
    C = np.random.randint(0,10,size=(20,20))
    print(C)
    # D,_ = ref(C)
    # print(_)
    # print(D)
    # print(rref(C))
    s1 = time.time()
    det1 = np.linalg.det(C)
    e1 = time.time()
    print(f'Determinant numpy.linalg = {det1} \t Time = {e1-s1}')
    s2 = time.time()
    det2 = det_from_ref(C)
    e2 = time.time()
    print(f'Determinant det_from_ref = {det2} \t Time = {e2-s2}')
    # print(det(C,C.shape[0],datatype=float))