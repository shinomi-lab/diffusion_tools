import difftools.diffusion.ic as ddi
import difftools.diffusion.multi._jit as ddm
import difftools.maximization as dm
import difftools.algebra as da
import difftools.trial as dt

if __name__ == "__main__":
    for p in [ddi, ddm, dm, da, dt]:
        if hasattr(p, "cc"):
            p.cc.compile()