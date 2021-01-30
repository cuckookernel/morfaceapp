"""A cli interface to the face combination algo"""

import sys
from pathlib import Path

import PIL.Image
# import face_landmark_experiments.fld as fld
# import face_landmark_experiments.face_comb as fcomb
from backend import fld
import backend.face_comb as fcomb

model = fld.load_landmarks_model()
retinaface = fld.Retinaface.Retinaface()


def main():
    # %%
    assert len(sys.argv) == 4

    fname1 = sys.argv[1]
    fname2 = sys.argv[2]
    out_name = sys.argv[3]
    # %%
    img1_raw = PIL.Image.open( fname1 )
    print( f"Loaded {fname1} - image of size: {img1_raw.size}")

    img2_raw = PIL.Image.open(fname2)
    print(f"Loaded {fname2} - image of size: {img2_raw.size}")

    pimg1 = fcomb.landmarks_1_face( img1_raw, model, retinaface )
    pimg2 = fcomb.landmarks_1_face( img2_raw, model, retinaface )

    out_path = Path(fname1).parent / (out_name + '.gif')
    print(f"\nfinal output saved on: {out_path}" )
    fcomb.gen_animation(pimg1, pimg2, out_path)


if __name__ == "__main__":
    main()
