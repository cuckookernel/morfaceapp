"""logic behind combine faces endpoint"""

from typing import Dict

import numpy as np
from backend.common import Request, get_image, put_img
from backend.face_comb import ProcessedImage, combine
from face_landmark_experiments.util import BBox

# %%


async def combine_faces( request: Request ) -> Dict:
    """Backend for combining faces request json data should contain
    img1_key, img2_key, face_bbox1, face_bbox2, landmarks1, landmarks2
    """
    req_json = await request.json()
    pimg1 = _make_processed_img( req_json, 1 )
    pimg2 = _make_processed_img( req_json, 2 )
    lambd = req_json['lambd']

    out_img = combine(pimg1, pimg2, lambd )
    img_key = put_img( out_img )

    return dict(img_key=img_key)


def _make_processed_img( req_json: Dict, idx: int ) -> ProcessedImage:

    img_key = req_json[f'img{idx}_key']
    img = get_image( img_key )
    bbox_dict = req_json[f'face_bbox{idx}']
    bbox = BBox( [bbox_dict['left'], bbox_dict['right'], bbox_dict['top'], bbox_dict['bottom']] )
    landmarks = np.array(req_json[f'landmarks{idx}'])

    return ProcessedImage(img=img, bbox=bbox, landmarks=landmarks)
