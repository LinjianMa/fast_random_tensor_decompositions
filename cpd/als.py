import time
import numpy as np
from als.als_optimizer import DTALS_base, ALS_leverage_base
from backend import numpy_ext
from .common_kernels import sub_lists, mttkrp, khatri_rao_product_chain


class CP_leverage_Optimizer(ALS_leverage_base):
    def __init__(self, tenpy, T, A, args):
        ALS_leverage_base.__init__(self, tenpy, T, A, args)

    def _solve(self, lhs, rhs, k):
        self.A[k] = self.tenpy.solve(
            self.tenpy.transpose(lhs) @ lhs,
            self.tenpy.transpose(lhs) @ rhs)

    def _form_lhs(self, list_a):
        out = self.tenpy.ones(list_a[0].shape)
        for a in list_a:
            out *= a
        return out


class CP_DTALS_Optimizer(DTALS_base):
    def __init__(self, tenpy, T, A):
        DTALS_base.__init__(self, tenpy, T, A)
        self.ATA_hash = {}
        for i in range(len(A)):
            self.ATA_hash[i] = tenpy.dot(A[i], tenpy.transpose(A[i]))

    def _einstr_builder(self, M, s, ii):
        ci = ""
        nd = M.ndim
        if len(s) != 1:
            ci = "R"
            nd = M.ndim - 1

        str1 = ci + "".join([chr(ord('a') + j) for j in range(nd)])
        str2 = "R" + (chr(ord('a') + ii))
        str3 = "R" + "".join([chr(ord('a') + j) for j in range(nd) if j != ii])
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def compute_lin_sys(self, i, Regu):
        S = None
        for j in range(len(self.A)):
            if j != i:
                if S is None:
                    S = self.ATA_hash[j].copy()
                else:
                    S *= self.ATA_hash[j]
        S += Regu * self.tenpy.eye(S.shape[0])
        return S

    def _solve(self, i, Regu, s):
        new_Ai = self.tenpy.solve(self.compute_lin_sys(i, Regu), s)
        self.ATA_hash[i] = self.tenpy.dot(new_Ai, self.tenpy.transpose(new_Ai))
        if i == self.order - 1:
            self.mttkrp_last_mode = s
        return new_Ai
