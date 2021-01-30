"""definitions common to many moduless"""

import logging
from pathlib import Path
from hashlib import sha256

from flask.wrappers import Request

import numpy as np
import PIL.Image

Array = np.ndarray

TMP_IMAGES_PATH = Path( "/var/tmp/images" )
TMP_IMAGES_PATH.mkdir( parents=True, exist_ok=True )
# %%


def _setup_logging():
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='/tmp/myapp.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def get_image( img_key: str ) -> Array:
    """get previously uploaded image by key"""
    fpath = TMP_IMAGES_PATH / img_key
    img_raw = PIL.Image.open( fpath )
    img = np.asarray( img_raw )

    return img
    # %%


def put_img( img: PIL.Image, prefix: str = "out" ) -> str:
    """put image and return key"""
    img_key = prefix + "_" + sha256( np.asarray(img).tobytes() ).hexdigest()[:24] + ".png"
    fpath = TMP_IMAGES_PATH / img_key
    img.save( fpath )

    return img_key
    # %%


_setup_logging()

l_info = logging.info
