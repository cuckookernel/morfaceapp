"""Backend to upload files"""

from hashlib import sha256

from flask import Response
from .common import Request, TMP_IMAGES_PATH


async def upload_image( request: Request ):
    """upload image via post request"""
    print( f"upload_image: headers={request.headers}" )
    content_length = int(request.headers['Content-Length'])
    if content_length > 1024 ** 2:
        raise RuntimeError(f"data is too big: {content_length} bytes")

    # image_bytes = await request.body()
    image_bytes = request.get_data()
    print( f"upload_image: body length= {len(image_bytes)}" )

    sha = sha256(image_bytes).hexdigest()

    img_ext = request.headers['Content-Type'].split('/')[1]

    img_key = (sha[:16] + "." + img_ext)
    fpath = TMP_IMAGES_PATH / img_key

    with fpath.open("wb") as f_out:
        f_out.write(image_bytes)

    return { "img_key": img_key }


async def download_image( request: Request ):
    """upload image via post request"""

    # image_bytes = await request.body()
    img_key = request.args['img_key']

    print( f"download_image: {img_key}" )

    fpath = TMP_IMAGES_PATH / img_key

    with fpath.open("rb") as f_out:
        img_bytes = f_out.read()

    return Response(img_bytes)
