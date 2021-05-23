import numpy as np
import time
import abc, six
import collections
from tucker.common_kernels import kron_products, matricize_tensor, count_sketch
try:
    import Queue as queue
except ImportError:
    import queue


@six.add_metaclass(abc.ABCMeta)
class DTALS_base():
    def __init__(self, tenpy, T, A):
        self.tenpy = tenpy
        self.T = T
        self.order = len(T.shape)
        self.A = A
        self.R = A[0].shape[1]
        self.num_iters_map = {"dt": 0, "ppinit": 0, "ppapprox": 0}
        self.time_map = {"dt": 0., "ppinit": 0., "ppapprox": 0.}
        self.pp_init_iter = 0

    @abc.abstractmethod
    def _einstr_builder(self, M, s, ii):
        return

    @abc.abstractmethod
    def _solve(self, i, Regu, s):
        return

    def step(self, Regu=1e-6):
        self.num_iters_map["dt"] += 1
        t0 = time.time()

        q = queue.Queue()
        for i in range(len(self.A)):
            q.put(i)
        s = [(list(range(len(self.A))), self.T)]
        while not q.empty():
            i = q.get()
            while i not in s[-1][0]:
                s.pop()
                assert (len(s) >= 1)
            while len(s[-1][0]) != 1:
                M = s[-1][1]
                idx = s[-1][0].index(i)
                ii = len(s[-1][0]) - 1
                if idx == len(s[-1][0]) - 1:
                    ii = len(s[-1][0]) - 2

                einstr = self._einstr_builder(M, s, ii)
                N = self.tenpy.einsum(einstr, M, self.A[ii])

                ss = s[-1][0][:]
                ss.remove(ii)
                s.append((ss, N))
            self.A[i] = self._solve(i, Regu, s[-1][1])

        dt = time.time() - t0
        self.time_map["dt"] = (self.time_map["dt"] *
                               (self.num_iters_map["dt"] - 1) +
                               dt) / self.num_iters_map["dt"]
        return self.A


def leverage_scores(tenpy, A):
    """
    Leverage scores of the matrix A
    """
    q, _ = tenpy.qr(A.T)
    return np.asarray([q[i, :] @ q[i, :] for i in range(q.shape[0])])


@six.add_metaclass(abc.ABCMeta)
class ALS_leverage_base():
    def __init__(self, tenpy, T, A, args):
        self.tenpy = tenpy
        self.T = T
        self.order = len(T.shape)
        self.A = A
        self.R = A[0].shape[0]
        self.epsilon = args.epsilon
        self.outer_iter = args.outer_iter
        self.fix_percentage = args.fix_percentage
        self.p_distributions = [
            leverage_scores(self.tenpy, self.A[i]) / self.R
            for i in range(self.order)
        ]
        assert self.order == 3
        self.sample_size_per_mode = int(self.R / self.epsilon)
        self.sample_size = self.sample_size_per_mode * self.sample_size_per_mode
        tenpy.printf(
            f"Leverage sample size is {self.sample_size}, rank is {self.R}")

    @abc.abstractmethod
    def _solve(self, lhs, rhs, k):
        return

    @abc.abstractmethod
    def _form_lhs(self, list_a):
        return

    def sample_krp_leverage(self, k):
        idx = [None for _ in range(self.order)]
        weights = [1. for _ in range(self.sample_size)]
        indices = [i for i in range(self.order) if i != k]
        for i, v in enumerate(indices):
            if self.fix_percentage == 0.:
                idx_one_mode = [
                    np.random.choice(np.arange(self.T.shape[v]),
                                     p=self.p_distributions[v])
                    for _ in range(self.sample_size)
                ]
                weights = [
                    weights[j] * self.p_distributions[v][idx_one_mode[j]]
                    for j in range(self.sample_size)
                ]
                idx[v] = idx_one_mode
            else:
                # deterministic sampling
                idx_one_mode = np.asarray(self.p_distributions[v]).argsort(
                )[len(self.p_distributions[v]) -
                  self.sample_size_per_mode:][::-1]
                if i == 0:
                    idx[v] = kron_products(
                        [np.ones(self.sample_size_per_mode),
                         idx_one_mode]).astype('int')
                elif i == 1:
                    idx[v] = kron_products(
                        [idx_one_mode,
                         np.ones(self.sample_size_per_mode)]).astype('int')
                else:
                    raise NotImplementedError

        assert len(idx) == self.order
        if self.fix_percentage == 0.:
            weights = 1. / (np.sqrt(self.sample_size * np.asarray(weights)))
        else:
            weights = [1. for _ in range(self.sample_size)]
        return idx, weights

    def lhs_sample(self, k, idx, weights):
        # form the krp or kronecker product
        lhs = []
        for s_i in range(self.sample_size):
            list_a = []
            for j in range(self.order):
                if j == k:
                    continue
                list_a.append(self.A[j][:, idx[j][s_i]])
            lhs.append(self._form_lhs(list_a) * weights[s_i])
        # TODO: change this to general tenpy?
        return np.asarray(lhs)

    def rhs_sample(self, k, idx, weights):
        # sample the tensor
        rhs = []
        for s_i in range(self.sample_size):
            sample_idx = [idx[j][s_i] for j in range(k)]
            sample_idx += [slice(None)]
            sample_idx += [idx[j][s_i] for j in range(k + 1, self.order)]
            rhs.append(self.T[tuple(sample_idx)] * weights[s_i])
        # TODO: change this to general tenpy?
        return np.asarray(rhs)

    def step(self, Regu=0):
        for l in range(self.outer_iter):
            for k in range(self.order):
                # get the sampling indices
                idx, weights = self.sample_krp_leverage(k)
                # get the sampled lhs and rhs
                lhs = self.lhs_sample(k, idx, weights)
                rhs = self.rhs_sample(k, idx, weights)
                self._solve(lhs, rhs, k)
                self.p_distributions[k] = leverage_scores(
                    self.tenpy, self.A[k]) / self.R
        return self.A


@six.add_metaclass(abc.ABCMeta)
class ALS_countsketch_base():
    def __init__(self, tenpy, T, A, args):
        self.tenpy = tenpy
        self.T = T
        self.order = len(T.shape)
        self.A = A
        self.R = A[0].shape[0]
        self.epsilon = args.epsilon
        self.outer_iter = args.outer_iter
        self.sample_size = int(self.R**(self.order - 1) / (self.epsilon**2))
        self.core_dims = args.hosvd_core_dim
        tenpy.printf(
            f"Countsketch sample size is {self.sample_size}, rank is {self.R}")
        self._build_matrices_embeddings()
        self._build_tensor_embeddings()

    def _build_matrices_embeddings(self):
        self.hashed_indices_factors = []
        self.rand_signs_factors = []
        for dim in range(self.order):
            indices = [i for i in range(dim)
                       ] + [i for i in range(dim + 1, self.order)]
            hashed_indices = [
                np.random.choice(self.sample_size,
                                 self.A[i].shape[1],
                                 replace=True) for i in indices
            ]
            rand_signs = [
                np.random.choice(2, self.A[i].shape[1], replace=True) * 2 - 1
                for i in indices
            ]
            self.hashed_indices_factors.append(hashed_indices)
            self.rand_signs_factors.append(rand_signs)

    def _build_tensor_embeddings(self):
        self.sketched_Ts = []
        for dim in range(self.order):
            hashed_indices = self.hashed_indices_factors[dim]
            rand_signs = self.rand_signs_factors[dim]

            signs_tensor = kron_products(rand_signs)
            indices_tensor = hashed_indices[-1]
            for index in reversed(hashed_indices[:-1]):
                new_indices = np.zeros((len(index) * len(indices_tensor), ))
                for j in range(len(index)):
                    new_indices[j * len(indices_tensor):(j + 1) *
                                len(indices_tensor)] = np.mod(
                                    indices_tensor + index[j],
                                    self.sample_size)
                indices_tensor = new_indices
            assert indices_tensor.shape == signs_tensor.shape

            reshape_T = matricize_tensor(self.tenpy, self.T, dim)
            sketched_mat_T = count_sketch(reshape_T,
                                          self.sample_size,
                                          hashed_indices=indices_tensor,
                                          rand_signs=signs_tensor)
            assert sketched_mat_T.shape == (self.T.shape[dim],
                                            self.sample_size)
            self.sketched_Ts.append(sketched_mat_T.transpose())

    @abc.abstractmethod
    def _solve(self, lhs, rhs, k):
        return

    @abc.abstractmethod
    def _form_lhs(self, k):
        return

    def step(self, Regu=1e-7):
        for l in range(self.outer_iter):
            for k in range(self.order):
                lhs = self._form_lhs(k)
                self._solve(lhs, self.sketched_Ts[k], k)
        return self.A


@six.add_metaclass(abc.ABCMeta)
class ALS_countsketch_su_base(ALS_countsketch_base):
    def __init__(self, tenpy, T, A, args):
        ALS_countsketch_base.__init__(self, tenpy, T, A, args)
        self.sample_size_core = self.sample_size * self.R
        self.init = False

    def _build_embedding_core(self):
        indices = [i for i in range(self.order)]
        self.hashed_indices_core = [
            np.random.choice(self.sample_size_core,
                             self.A[i].shape[1],
                             replace=True) for i in indices
        ]
        self.rand_signs_core = [
            np.random.choice(2, self.A[i].shape[1], replace=True) * 2 - 1
            for i in indices
        ]

        signs_tensor = kron_products(self.rand_signs_core)
        indices_tensor = self.hashed_indices_core[-1]
        for index in reversed(self.hashed_indices_core[:-1]):
            new_indices = np.zeros((len(index) * len(indices_tensor), ))
            for j in range(len(index)):
                new_indices[j * len(indices_tensor):(j + 1) *
                            len(indices_tensor)] = np.mod(
                                indices_tensor + index[j],
                                self.sample_size_core)
            indices_tensor = new_indices
        assert indices_tensor.shape == signs_tensor.shape

        reshape_T = self.T.reshape((1, -1))
        # has shape 1 x s
        sketched_mat_T = count_sketch(reshape_T,
                                      self.sample_size_core,
                                      hashed_indices=indices_tensor,
                                      rand_signs=signs_tensor)
        assert sketched_mat_T.shape == (1, self.sample_size_core)
        self.sketched_T_core = sketched_mat_T.transpose()

    @abc.abstractmethod
    def _solve(self, lhs, rhs, k):
        return

    @abc.abstractmethod
    def _solve_core(self, lhs, rhs):
        return

    @abc.abstractmethod
    def _form_lhs(self, k):
        return

    @abc.abstractmethod
    def _form_lhs_core(self):
        return

    def step(self, Regu=1e-7):
        if self.init is False:
            self._build_embedding_core()
            self.init = True
        for l in range(self.outer_iter):
            for k in range(self.order):
                lhs = self._form_lhs(k)
                self._solve(lhs, self.sketched_Ts[k], k)
            lhs = self._form_lhs_core()
            self._solve_core(lhs, self.sketched_T_core)
        return self.A
