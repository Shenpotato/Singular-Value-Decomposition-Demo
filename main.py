import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
              [1, 1, 0, 1, 0, 0, 1, 1, 0, 2, 1],
              [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1]])
A = A.T

# (a) Decompose 𝐴 using Singular Value Decomposition, i.e., 𝐴 = 𝑈𝑆𝑉𝑇
u, s, v = np.linalg.svd(A)
s = np.diag(s)
temp = np.zeros((3, 8))
s = np.c_[s, temp].T

q = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]])

# (b)Given the query 𝑞 = [0 0 0 0 0 1 0 0 0 1 1],
# compute the inner product scores of 𝑑1, 𝑑2 and 𝑑3 using the decomposed matrices.
# Verify if the result is the same as performing the dot product of 𝑞 and 𝐴.
result1 = np.dot(q, A)

temp2 = np.dot(np.dot(u, s), v)
result2 = np.dot(q, temp2)

# (c)Apply Rank-2 approximation to the decomposed matrices, i.e., give 𝑈2, 𝑆2 and 𝑉2, 𝑉2 ,
# and 𝐴2 (The subscript “2” means Rank-2 approximation)
u2 = u[:,0:2]
s2 = s[0:2,0:2]
v2 = v[0:2,:]

A2 = np.dot(np.dot(u2,s2),v2)

# (d) Obtain the document vectors and query vector 𝑞 = [0 0 0 0 0 1 0 0 0 1 1] in the reduced
# 2-dimensional space, and plot them in 2-D coordinates.
result3 = np.dot(s2,v2)
plt.scatter(result3[0],result3[1])
plt.show()

# (f) Compute inner product scores of d1, d2 and d3 to q after Rank-2 approximation.
