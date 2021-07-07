import numpy as np
import numba
import numba.types as nt
import numba.typed as ntd

InfoType_F = 0
InfoType_T = 1
InfoTypes_n = 2

@numba.njit
def make_InfoTypes() -> np.ndarray:
    a = np.zeros(InfoTypes_n, dtype=np.int64)
    for it in [InfoType_F, InfoType_T]:
      a[it] = it

    return a
