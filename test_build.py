import difftools.diffusion as dd
import difftools.maximization as dm

if __name__ == "__main__":
    if hasattr(dd, "cc"):
        dd.cc.compile()
    if hasattr(dm, "cc"):
        dm.cc.compile()
