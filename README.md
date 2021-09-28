# minilini

A very (very) small linear algebra library providing step-by-step solutions to various problems. I created this program because I needed a way to quickly generate step-by-step linear algebra problem solutions for a class that I was teaching. This is the result. Before we begin, we should mention that the required standard python libraries are `time`, `numpy`, `nptyping`, `sympy`, `fractions`, and `typing`.

## Step-by-Step Solutions for the following types of problems:

1. Row Echelon Form (`ref`)
2. Reduced Row Echelon Form (`rref`)
3. Determinant (`det`)

## Row Echelon Form

Finding the row echelon form of a matrix is one of the most essential skills in introductory linear algebra. For an ordinary matrix, that is, a matrix that is not the augmented matrix of a system of equations, the row echelon form can tell us the rank and determinant of a matrix as well as whether the columns of the matrix are linearly independent or not. For an augemented matrix, one can tell whether the associated system is inconsistent or not.

### Basic Example

Suppose we have the following matrix

![equation](https://latex.codecogs.com/gif.latex?A%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%202%20%26%201%20%5C%5C%202%20%26%204%20%26%208%20%5C%5C%201%20%26%201%20%26%204%20%5Cend%7Bbmatrix%7D.)

We can represent this matrix in python as

```python
import numpy as np
A = np.array([[1,2,1],[2,4,8],[1,1,4]])
```

The function `ref` takes any numpy matrix as a parameter and returns two different things. Firstly, it returns the row echelon form of the input matrix. Secondly, it returns the scaling factors that were used in the Gaussian elimination algorithm to obtain the row echelon form. Thus, we can use the function as follows:

```python
import minilini as ml
refA, scalingFactorsA = ml.ref(A)
```

Printing our output, we find that 

![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BrefA%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%202%20%26%201%20%5C%5C%200%20%26%201%20%26%20-3%20%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D)

and

![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BscalingFactorsA%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%20-1%20%26%20-1%20%26%206%20%5Cend%7Bbmatrix%7D.)

### Advanced Example

Let's take the same matrix as in the Basic Example above

![equation](https://latex.codecogs.com/gif.latex?A%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%202%20%26%201%20%5C%5C%202%20%26%204%20%26%208%20%5C%5C%201%20%26%201%20%26%204%20%5Cend%7Bbmatrix%7D.)

Again, we can represent this matrix in python as before,

```python
import numpy as np
A = np.array([[1,2,1],[2,4,8],[1,1,4]])
```

Now, instead of calling the row echelon form function as before, we will add an optional parameter so that the function prints the steps to obtaining the solution

```python
import minilini as ml
refA, scalingFactorsA = ml.ref(A, steps=True)
```

Running this code results in the following text being returned

```python
Subtract 2.0 * Row 1 from Row 2.
[[1. 2. 1.]
 [0. 0. 6.]
 [1. 1. 4.]]


Subtract 1.0 * Row 1 from Row 3.
[[ 1.  2.  1.]
 [ 0.  0.  6.]
 [ 0. -1.  3.]]


Swap Row 2 and Row 3.
[[ 1.  2.  1.]
 [ 0. -1.  3.]
 [ 0.  0.  6.]]


Scale Row 2 by -1.0.
[[ 1.  2.  1.]
 [ 0.  1. -3.]
 [ 0.  0.  6.]]


Scale Row 3 by 6.0.
[[ 1.  2.  1.]
 [ 0.  1. -3.]
 [ 0.  0.  1.]]
```

This comprises a step-by-step solution to the problem of finding the row echelon form of the matrix ![equation](https://latex.codecogs.com/gif.latex?A). From this, we can see that the rank of the matrix is 3, the determinant is 6, the associated system is inconsistent, etc.

## Reduced Row Echelon Form

The reduced row echelon form of a matrix is a natural extension to the process of finding the row echelon form. For one, it allows us to determine whether a matrix is invertible or not. The usage of `rref` is very similar to that of the `ref` function. 

### Example

Using the same matrix ![equation](https://latex.codecogs.com/gif.latex?A) as above and initializing the matrix in python, we use `rref` as follows

```python
import minilini as ml
rrefA = ml.rref(A, steps=True)
```

and obtain the following output in addition to the output from `ref` above

```python
Subtract 2.0 * Row 2 from Row 1
[[ 1.  0.  7.]
 [ 0.  1. -3.]
 [ 0.  0.  1.]]
 
Subtract 7.0 * Row 3 from Row 1.
[[ 1.  0.  0.]
 [ 0.  1. -3.]
 [ 0.  0.  1.]]
 
Subtract -3.0 * Row 3 from Row 2.
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

## Information

The information function gives one everything that someone in an introductory linear algebra class would want to know about a matrix. The information function is structured as follows

```python
def information(
    matrix: npt.NDArray[Any,Any],
    verbose: bool = True,
    augmented: bool = False
) -> None:
```

with one required argument, a matrix, and two optional arguments. The verbose argument is default set to `True`. If one would like less information about the input matrix, set `verbose = False`. The augmented argument is also default set to `False`. If the input matrix is the augmented matrix representing a system of equations, set `augmented = True`. 

Let's again, let ![equation](https://latex.codecogs.com/gif.latex?A) be the matrix from before, then running

```python
import minilini as ml
info = information(A)
```

and printing the info yields

```python
print(info)
```

```python
Information about the Matrix:

[[1 2 1]
 [2 4 8]
 [1 1 4]]

Row Echelon Form:

[[ 1.  2.  1.]
 [ 0.  1. -3.]
 [ 0.  0.  1.]]

Reduced Row Echelon Form:

[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

Rows:            3
Columns:         3
Rank:            3
Nullity:         0
Invertible:      True
Determinant:     6.0
Eigenvalues:     [7.67+0.j   0.67+0.58j 0.67-0.58j]
Basis of Col:    [[1, 2, 1], [2, 4, 1], [1, 8, 4]]
Free Vars:       []
Columns of A are linearly INDEPENDENT.
The associated system of 3 equations in 3 variables is CONSISTENT.
The associated system is EXACTLY DETERMINED.
```

If we instead set `augmented = True`, a few things about the information will change

```python
import minilini as ml
info = information(A, augmented = True)
```

and printing the info yields

```python
print(info)
```

```python
Information about the Matrix:

[[1 2 1]
 [2 4 8]
 [1 1 4]]

Row Echelon Form:

[[ 1.  2.  1.]
 [ 0.  1. -3.]
 [ 0.  0.  1.]]

Reduced Row Echelon Form:

[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

Rows:            3
Columns:         3
Rank:            3
Nullity:         0
Invertible:      True
Determinant:     6.0
Eigenvalues:     [7.67+0.j   0.67+0.58j 0.67-0.58j]
Basis of Col:    [[1, 2, 1], [2, 4, 1], [1, 8, 4]]
Free Vars:       []
Columns of A are linearly INDEPENDENT.
The associated system of 3 equations in 2 variables is INCONSISTENT.
The associated system is OVERDETERMINED.
```

The reason for this is because, if this is not an augmented matrix, the associated system is 

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Barray%7D%7Bl%7D%20x%20&plus;%202y%20&plus;%20z%20%3D%200%20%5C%5C%202x%20&plus;%204y%20&plus;%208z%20%3D%200%20%5C%5C%20x%20&plus;%20y%20&plus;%204z%20%3D%200%20%5Cend%7Barray%7D,)

and if this is an augmented matrix, the associated system is 

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Barray%7D%7Bl%7D%20x%20&plus;%202y%20%3D%201%20%5C%5C%202x%20&plus;%204y%20%3D%208%20%5C%5C%20x%20&plus;%20y%20%3D%204%20%5Cend%7Barray%7D.)


## Determinant

The determinant of a matrix is a scalar associated with every square matrix from which various properties of a matrix can be determined. For example, a matrix is invertable if and only if the determinant of that matrix is not zero. 

### Example

Working with our same sample matrix ![equation](https://latex.codecogs.com/gif.latex?A), we can calculate the determinant of this matrix using the cofactor expansion method with the function `det_steps`. Executing the code

```python
ml.det_steps(A)
```

we obtain

![equation](https://latex.codecogs.com/gif.latex?%281%29%20*%20%28%28%284%29%20*%20%284%29%29%20-%20%28%288%29%20*%20%281%29%29%29%29%20-%20%28%282%29%20*%20%28%28%282%29%20*%20%284%29%29%20-%20%28%288%29%20*%20%281%29%29%29%29%20&plus;%20%28%281%29%20*%20%28%28%282%29%20*%20%281%29%29%20-%20%28%284%29%20*%20%281%29%29%29%29.)
