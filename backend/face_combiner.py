
from typing import Dict
import json
from .common import Request, get_image, Array, put_img
from .face_comb import ProcessedImage, combine

import PIL.Image


def combine_faces( request: Request ) -> Dict:
    """backend for combining faces request json data should contain
    img1_key, img2_key, face_bbox1, face_bbox2, landmarks1, landmarks2"""
    req_json = json.loads( request.get_data() )
    pimg1 = _make_processed_img( req_json, 1 )
    pimg2 = _make_processed_img( req_json, 2 )
    lambd = req_json['lambd']

    out_img = combine(pimg1, pimg2, lambd )
    img_key = put_img( out_img )

    return dict(img_key=img_key)


def _make_processed_img( req_json: Dict, idx: int ) -> ProcessedImage:

    img_key = req_json[f'img{idx}_key']
    img = get_image( img_key )
    bbox = json.loads(req_json[f'face_bbox{idx}'])
    landmarks = Array(req_json[f'landmarks{idx}'])

    return ProcessedImage(img=img, bbox=bbox, landmarks=landmarks)
