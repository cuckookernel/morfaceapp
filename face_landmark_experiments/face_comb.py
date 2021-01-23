"""face combination"""
from typing import List
import time
from pathlib import Path
from dataclasses import dataclass
import PIL
from PIL import Image
import PIL.Image
import numpy as np

import fld
from fld import TRIANGLES
from util import BBox

Array = np.ndarray


@dataclass
class ProcessedImage:
    """An image for which landmarks have already been computeds"""
    img: Array  # shape: (h, w, c)
    bbox: BBox
    landmarks: Array


def landmarks_1_face(img0: PIL.Image, model, retinaface):
    img = np.asarray(img0)

    faces = retinaface(img)
    landmarks, new_bbox = fld.process_1_face(faces[0], img, model)

    return ProcessedImage(img, new_bbox, landmarks)


def draw_landmarks( pimg: ProcessedImage ) -> PIL.Image:
    img = drawLandmark_multiple( pimg.img, pimg.bbox, pimg.landmarks )
    return Image.fromarray( img )


def gen_animation(pimg1: ProcessedImage, pimg2: ProcessedImage,
                  out_fpath: Path, lambdas: List[float] = None):
    if lambdas is None:
        lambdas = [0.5, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95,
                   1.0,
                   0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.66, 0.64, 0.62, 0.6, 0.58, 0.56, 0.54, 0.5,
                   0.48, 0.46, 0.44, 0.42, 0.4, 0.38, 0.36, 0.34, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05,
                   0.0,
                   0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48,
                   0.5]
    imgs = []
    for i, lambdah in enumerate(lambdas):
        img = combine(pimg1, pimg2, lambdah)
        print(f"{i:2d}/{len(lambdas)}", end="\r")
        imgs.append(img)

    head, *tail = imgs

    format = out_fpath.suffix[1:].upper()
    print(f"format: {format}")

    head.save(fp=out_fpath, format=format, append_images=imgs,
              save_all=True, duration=200, loop=0)


def gen_transition( pimg1: ProcessedImage, pimg2: ProcessedImage, out_path: Path,
                    lambdas: List[float] = None ):

    if not out_path.exists():
        print( f'creating: {out_path}')
        out_path.mkdir( parents=True, exist_ok=True )
    else:
        print( f'saving results into existing: {out_path}')

    if lambdas is None:
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48,
                   0.5, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.70, 0.80, 0.9, 1.0]
    for lambdah in lambdas:
        img = combine(pimg1, pimg2, lambdah)
        img.save( out_path / f"c_{int(lambdah * 100) :02d}.png")


def combine(pimg1: ProcessedImage, pimg2: ProcessedImage, lambdah: float):
    lm_p1 = pimg1.bbox.projectLandmark(pimg1.landmarks)
    lm_p2 = pimg2.bbox.projectLandmark(pimg2.landmarks)

    lm_pc = (1 - lambdah) * lm_p1 + lambdah * lm_p2

    out_w, out_h = (300, 300)
    img_c = 255 * np.ones((out_h, out_w, 3), dtype=np.uint8)
    bbox = BBox([0, out_w, -50, -50 + out_w])

    lm_c = bbox.reprojectLandmark(lm_pc)

    img_combiner = ImageCombiner(pimg1, pimg2, lambdah)
    bcalc2 = BariCalc2(lm_c, dim=out_h)

    def do_col(col):
        x = col * np.ones((1, out_h))
        y = np.arange(out_h).reshape((1, out_h))

        row_indices, t_indices, b = bcalc2.triangle_indices_coords(x, y)
        pixelsc = img_combiner.combine(b, t_indices)

        img_c[(row_indices, col)] = pixelsc

    t1 = time.perf_counter()

    for col in range(out_w):
        do_col(col)

    t2 = time.perf_counter()
    # print( t2 - t1 )

    return PIL.Image.fromarray(img_c)


def get_triangle_vertex_coords(landmarks, vertex_idx):
    return np.array([landmarks[triangle[vertex_idx]] for triangle in TRIANGLES])


class ImageCombiner:

    def __init__(self, pimg1: ProcessedImage, pimg2: ProcessedImage, lambdah: float):
        self.img1 = pimg1.img
        self.img2 = pimg2.img
        self.lambdah = lambdah

        self.p10 = get_triangle_vertex_coords(pimg1.landmarks, 0)
        self.p11 = get_triangle_vertex_coords(pimg1.landmarks, 1)
        self.p12 = get_triangle_vertex_coords(pimg1.landmarks, 2)

        self.p20 = get_triangle_vertex_coords(pimg2.landmarks, 0)
        self.p21 = get_triangle_vertex_coords(pimg2.landmarks, 1)
        self.p22 = get_triangle_vertex_coords(pimg2.landmarks, 2)

    def combine(self, b: Array, t_indices: Array):
        # coords of points with baricentric coordinates b relative to triangles t_indices in img 1
        coords1 = ( b[:, [0]] * self.p10[t_indices] + b[:, [1]] * self.p11[t_indices]
                    + b[:, [2]] * self.p12[t_indices] )
        coords1 = coords1.astype(np.int32)

        # coords of points with baricentric coordinates b relative to triangles t_indices in img 2
        coords2 = ( b[:, [0]] * self.p20[t_indices] + b[:, [1]] * self.p21[t_indices]
                    + b[:, [2]] * self.p22[t_indices] )
        coords2 = coords2.astype(np.int32)

        # extract pixel values from img1 at coords1
        pixels1 = self.img1[(coords1[:, 1], coords1[:, 0])]
        pixels2 = self.img2[(coords2[:, 1], coords2[:, 0])]

        pixelsc = ((1.0 - self.lambdah) * pixels1 + self.lambdah * pixels2).astype(np.uint8)

        return pixelsc


class BariCalc2:
    """baricentric coordinates calculation"""
    def __init__(self, lm_c: Array, dim: int):
        self.lm_c = lm_c
        self.find_tri_cnt = 0

        self.dim = dim
        self.num_tri = len(TRIANGLES)
        p1 = get_triangle_vertex_coords( lm_c, 0)
        p2 = get_triangle_vertex_coords( lm_c, 1)
        p3 = get_triangle_vertex_coords( lm_c, 2)
        # p1 = np.array([lm_c[triangle[0]] for triangle in self.triangles])
        # p2 = np.array([lm_c[triangle[1]] for triangle in self.triangles])
        # p3 = np.array([lm_c[triangle[2]] for triangle in self.triangles])

        self.x1 = np.tile(p1[:, [0]], (1, dim))
        self.y1 = np.tile(p1[:, [1]], (1, dim))
        self.dx2 = np.tile(p2[:, [0]] - p1[:, [0]], (1, dim))
        self.dy2 = np.tile(p2[:, [1]] - p1[:, [1]], (1, dim))
        self.dx3 = np.tile(p3[:, [0]] - p1[:, [0]], (1, dim))
        self.dy3 = np.tile(p3[:, [1]] - p1[:, [1]], (1, dim))

    def coords_triangles(self, x: Array, y: Array) -> Array:
        """baricentric coordinates for all triangles"""
        assert x.shape == (1, self.dim)
        assert y.shape == (1, self.dim)
        dx = np.tile(x, (self.num_tri, 1)) - self.x1
        dy = np.tile(y, (self.num_tri, 1)) - self.y1

        det = (self.dy2 * self.dx3 - self.dx2 * self.dy3)
        b2 = (+ dy * self.dx3 - dx * self.dy3) / det
        b3 = (- dy * self.dx2 + dx * self.dy2) / det
        b1 = 1.0 - b2 - b3

        return np.array([b1, b2, b3]).transpose((2, 1, 0))

    def triangle_indices_coords(self, x: Array, y: Array):
        b_coords = self.coords_triangles(x, y)
        all_pos = (b_coords[:, :, 0] >= 0.) & (b_coords[:, :, 1] >= 0.) & (b_coords[:, :, 2] >= 0)

        non_zero = np.nonzero(all_pos)
        row_indices = non_zero[0]
        t_indices = non_zero[1]
        b_coords_ret = b_coords[non_zero]

        return row_indices, t_indices, b_coords_ret
