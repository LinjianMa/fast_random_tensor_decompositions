import numpy as np
import queue
import scipy
from .common_kernels import n_mode_eigendec, kron_products, count_sketch, matricize_tensor, one_mode_solve
from cpd.common_kernels import krp
from als.als_optimizer import DTALS_base, ALS_leverage_base, ALS_countsketch_base, ALS_countsketch_su_base


class Tucker_leverage_Optimizer(ALS_leverage_base):
    def __init__(self, tenpy, T, A, args):
        ALS_leverage_base.__init__(self, tenpy, T, A, args)
        self.core_dims = args.hosvd_core_dim
        self.core = tenpy.random(self.core_dims)

    def _solve(self, lhs, rhs, k):
        self.A[k], self.core = one_mode_solve(self.tenpy, lhs, rhs, self.R, k,
                                              self.core_dims, self.order)

    def _form_lhs(self, list_a):
        return kron_products(list_a)


def kronecker_tensorsketch(tenpy, A, indices, sample_size, hashed_indices,
                           rand_signs):
    assert len(indices) == len(hashed_indices)
    # each A has size R x s
    sketched_A = [
        count_sketch(A[indices[i]],
                     sample_size,
                     hashed_indices=hashed_indices[i],
                     rand_signs=rand_signs[i]) for i in range(len(indices))
    ]
    # each A has size s x R
    sketched_A = [np.fft.fft(A.transpose(), axis=0) for A in sketched_A]
    # krp_A has size s x R^N
    krp_A = krp(tenpy, sketched_A).reshape((sample_size, -1))
    return np.real(np.fft.ifft(krp_A, axis=0))


class Tucker_countsketch_Optimizer(ALS_countsketch_base):
    def __init__(self, tenpy, T, A, args):
        ALS_countsketch_base.__init__(self, tenpy, T, A, args)
        self.core = tenpy.random(self.core_dims)

    def _solve(self, lhs, rhs, k):
        self.A[k], self.core = one_mode_solve(self.tenpy, lhs, rhs, self.R, k,
                                              self.core_dims, self.order)

    def _form_lhs(self, k):
        indices = [i for i in range(k)] + [i for i in range(k + 1, self.order)]
        return kronecker_tensorsketch(self.tenpy, self.A, indices,
                                      self.sample_size,
                                      self.hashed_indices_factors[k],
                                      self.rand_signs_factors[k])


class Tucker_countsketch_su_Optimizer(ALS_countsketch_su_base):
    def __init__(self, tenpy, T, A, args):
        ALS_countsketch_su_base.__init__(self, tenpy, T, A, args)
        self.core = tenpy.random(self.core_dims)

    def _solve(self, lhs, rhs, k):
        core_reshape = matricize_tensor(self.tenpy, self.core, k).transpose()
        lhs = lhs @ core_reshape
        mat = self.tenpy.solve(
            self.tenpy.transpose(lhs) @ lhs,
            self.tenpy.transpose(lhs) @ rhs)
        q, _ = self.tenpy.qr(mat.transpose())
        self.A[k] = q.transpose()

    def _solve_core(self, lhs, rhs):
        core_vec = self.tenpy.solve(
            self.tenpy.transpose(lhs) @ lhs,
            self.tenpy.transpose(lhs) @ rhs)
        self.core = core_vec.reshape(self.core_dims)

    def _form_lhs(self, k):
        indices = [i for i in range(k)] + [i for i in range(k + 1, self.order)]
        return kronecker_tensorsketch(self.tenpy, self.A, indices,
                                      self.sample_size,
                                      self.hashed_indices_factors[k],
                                      self.rand_signs_factors[k])

    def _form_lhs_core(self):
        indices = [i for i in range(self.order)]
        return kronecker_tensorsketch(self.tenpy, self.A, indices,
                                      self.sample_size_core,
                                      self.hashed_indices_core,
                                      self.rand_signs_core)


class Tucker_DTALS_Optimizer(DTALS_base):
    def __init__(self, tenpy, T, A):
        DTALS_base.__init__(self, tenpy, T, A)
        self.tucker_rank = []
        for i in range(len(A)):
            self.tucker_rank.append(A[i].shape[0])
        self.core = tenpy.ones([Ai.shape[0] for Ai in A])

    def _einstr_builder(self, M, s, ii):
        nd = M.ndim

        str1 = "".join([chr(ord('a') + j) for j in range(nd)])
        str2 = "R" + (chr(ord('a') + ii))
        str3 = "".join([chr(ord('a') + j) for j in range(ii)]) + "R" + "".join(
            [chr(ord('a') + j) for j in range(ii + 1, nd)])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _solve(self, i, Regu, s):
        # NOTE: Regu is not used here
        output = n_mode_eigendec(self.tenpy,
                                 s,
                                 i,
                                 rank=self.tucker_rank[i],
                                 do_flipsign=True)
        if i == len(self.A) - 1:
            str1 = "".join([chr(ord('a') + j) for j in range(self.T.ndim)])
            str2 = "R" + (chr(ord('a') + self.T.ndim - 1))
            str3 = "".join([chr(ord('a') + j)
                            for j in range(self.T.ndim - 1)]) + "R"
            einstr = str1 + "," + str2 + "->" + str3
            self.core = self.tenpy.einsum(einstr, s, output)
        return output
