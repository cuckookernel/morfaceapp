"""detect faces in a picture"""
import PIL.Image
import numpy as numpy
from backend.retinaface.core import Retinaface

from .common import Request, TMP_IMAGES_PATH, l_info, get_image
# %%


class FaceDetector:
    """detect all faces in an image"""
    def __init__(self):
        print( "FaceDetector building retinaface model" )
        self.retinaface = Retinaface()
        self.count = 0

    def __call__(self, request: Request ):
        img_key = request.query_params['img_key']
        img = get_image( img_key )

        face_bboxes0 = self.retinaface( img )
        #  need to convert to regular floats as np.float32 types 'is not json serializable'...
        face_bboxes = [ [float(x) for x in bbox] for bbox in face_bboxes0 ]
        print( f"Face detector: {face_bboxes}")

        return dict(img_key=img_key, face_bboxes=face_bboxes)
