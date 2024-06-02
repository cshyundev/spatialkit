import absl.app as app
import cv_utils.core.operations.hybrid_operations as hop
import cv_utils.core.operations.hybrid_math as hm
import numpy as np

def main(unused_args):

    M = np.array([[1, 0, 0, 0, 2],
                  [0, 0, 3, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 2, 0, 0, 0]], dtype=np.float32)

    u,s,vt = hm.svd(M)

    print(u.shape)
    print(s.shape)
    print(vt.shape)

    print(np.diag(s).shape)

    print(u @ np.diag(s) @ vt)


    return None

if __name__ == "__main__":
    app.run(main)