# uniformity check on the conv{(0,0),(0,4),(6,0)} triangle (see README.md)
# ~6 min on an RTX 5060 Ti (~150 dualGNN draws/s); scale `samples` to your hardware
from _diag import run

TRIANGLE = dict(name="conv{(0,0),(0,4),(6,0)}", tag="4x6tri",
                verts=[[0, 0], [0, 4], [6, 0]], N_FRT=405_706)

if __name__ == "__main__":
    run(TRIANGLE, samples=30_000)
