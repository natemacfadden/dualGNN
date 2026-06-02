# uniformity check on [0,4]^2 (see README.md). N ~ 735M, so this is under-powered
# at any practical sample count (it warns) -- a ~8 min (RTX 5060 Ti) smoke of the
# pipeline on the square; raise `samples` toward ~1e7 for the full paper regime.
from _diag import run

SQUARE = dict(name="[0,4]^2", tag="4x4sq",
              verts=[[0, 0], [0, 4], [4, 4], [4, 0]], N_FRT=735_430_548)

if __name__ == "__main__":
    run(SQUARE, samples=15_000)
