import numpy as np 


#define vectors 
v1 = np.array([2,3])
v2 = np.array([1,3])


# 


#vector  addition

add = v1 + v2 
print("Addition", add)

# scalar multiplication
scaled = 3 * v1
print("Scalar multiplication", scaled)



# 2️⃣ Dot Product (Used for similarity and projections)

dot = np.dot(v1, v2)  # (2×1) + (4×3) = 14
print("Dot Product:", dot)


# Define matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])

# Matrix addition
mat_add = A + B

# Matrix multiplication (not element-wise)
mat_mul = np.dot(A, B)

# Transpose
transpose = A.T

print("Matrix Addition:\n", mat_add)
print("Matrix Multiplication:\n", mat_mul)
print("Transpose of A:\n", transpose)
