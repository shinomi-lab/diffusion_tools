import difftools.diffusion as dd
import difftools.maximization as dm
import difftools.algebra as da
import difftools.trial as dt

if __name__ == "__main__":
    for p in [dd, dm, da, dt]:
        if hasattr(p, "cc"):
            p.cc.compile()