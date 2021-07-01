from typing import Tuple, Any, Set, List, Dict, Optional
import numpy as np
import numba
import numba.types as nt
import numba.typed as ntd
from ._types import *

from enum import Enum

S0 = 0
S1 = 1
S2 = 2
S3 = 3
S_n = S3 + 1

E_T0 = 0
E_F0 = 1
E_F1 = 2
E_T1 = 3
E_n = E_T1 + 1

def make_simple_stt() -> np.ndarray:
    a = np.zeros((S_n, InfoTypes_n, 2), np.int64)
    a[S0][InfoType_T] = [E_T0, S1]
    a[S0][InfoType_F] = [E_F0, S2]
    a[S1][InfoType_T] = [  -1, S1]
    a[S1][InfoType_F] = [E_F1, S3]
    a[S2][InfoType_T] = [E_T1, S3]
    a[S2][InfoType_F] = [  -1, S2]
    a[S3][InfoType_T] = [  -1, S3]
    a[S3][InfoType_F] = [  -1, S3]

    return a

def make_simple_assumption_map(pat_t0: np.float64, pat_f0: np.float64, pat_f1: np.float64, pat_t1: np.float64) -> np.ndarray:
    a = np.zeros(E_n, np.float64)
    a[E_T0] = pat_t0
    a[E_F0] = pat_f0
    a[E_F1] = pat_f1
    a[E_T1] = pat_t1

    return a

def make_simple_propagation_map(
    ppt_t0: np.float64, ppf_t0: np.float64, ppt_f0: np.float64, ppf_f0: np.float64,
    ppt_f1: np.float64, ppf_f1: np.float64, ppt_t1: np.float64, ppf_t1: np.float64
) -> np.ndarray:
    a = np.zeros((E_n, 2), np.float64)
    a[E_T0][InfoType_T] = ppt_t0
    a[E_T0][InfoType_F] = ppf_t0
    a[E_F0][InfoType_T] = ppt_f0
    a[E_F0][InfoType_F] = ppf_f0
    a[E_F1][InfoType_T] = ppt_f1
    a[E_F1][InfoType_F] = ppf_f1
    a[E_T1][InfoType_T] = ppt_t1
    a[E_T1][InfoType_F] = ppf_t1

    return a