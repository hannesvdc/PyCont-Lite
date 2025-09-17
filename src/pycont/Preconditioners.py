import scipy.sparse.linalg as slg

# returns LinearOperator y ≈ A^{-1} b using p_m(A)=alpha*sum_{j=0}^{m-1}(I-alpha A)^j
def polynomial_inverse(matvec, size, alpha, m) -> slg.LinearOperator:
    def apply(b):
        s = b.copy()
        vec = alpha*s
        for _ in range(1, m):
            s = s - alpha * matvec(s)
            vec = vec + alpha*s
        return vec
    return slg.LinearOperator((size, size), apply)