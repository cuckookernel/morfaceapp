
import numpy as np
from backend import fld

from .common import Request, get_image
from .retinaface.core import Retinaface


class LandMarksDetector:
    """detect all faces in an image"""

    def __init__(self):
        print( "Landmarks detector building retinaface model" )
        self.imgs_cache = {}
        self.retinaface = Retinaface()
        self.lm_detector = fld.load_landmarks_model()
        self.count = 0

    def __call__(self, request: Request ):
        img_key = request.query_params['img_key']
        face_idx = int(request.query_params['face_idx'])

        image = get_image(img_key)
        face_bbox = self.retinaface( image )[ face_idx ]

        landmarks, new_bbox = fld.process_1_face( face_bbox, image, self.lm_detector )

        return dict(img_key=img_key,
                    landmarks=np.round(landmarks, 2).tolist(),
                    new_bbox=new_bbox.to_dict())
