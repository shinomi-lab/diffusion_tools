import difftools.diffusion as dd
import difftools.maximization as dm
import difftools.algebra as da

if __name__ == "__main__":
    for p in [dd, dm, da]:
        if hasattr(p, "cc"):
            p.cc.compile()