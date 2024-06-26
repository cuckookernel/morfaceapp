"""setup server"""

import datetime as dt
import json
import logging
import time
from pathlib import Path

import backend.face_combiner as fc
import backend.face_detector as fd

# from starlette.requests import Request
# from starlette.middleware import Middleware
# from starlette.middleware.cors import CORSMiddleware
import backend.image_uploader as iul
import backend.landmarks_detector as lmd
import ray
import requests
from ray import serve

log = logging.getLogger()
log.setLevel(logging.INFO)
# %%


def _set_up_endpoints():
    # %%
    ray.init( num_cpus=7 )
    client = serve.start( http_host="0.0.0.0" )
    # %%
    # middleware = Middleware( CORSMiddleware, allow_origins=["*"], allow_methods=["*"] )
    # client = serve.start( http_middlewares=[ middleware ]  )
    # %%
    from importlib import reload
    reload( iul )
    reload( fd )
    reload( lmd )
    reload( fc )

    client.delete_endpoint("detect_faces")
    client.delete_endpoint("upload_image")
    client.delete_endpoint("detect_landmarks")
    client.delete_endpoint("combine_faces")

    client.delete_backend("face_detector")
    client.delete_backend("image_uploader")
    client.delete_backend("landmark_detector")
    client.delete_backend("face_combiner")
    # %%
    # Form a backend from our class and connect it to an endpoint.
    client.create_backend("face_detector", fd.FaceDetector)
    client.create_backend("image_uploader", iul.upload_image)

    client.create_backend( "landmark_detector", lmd.LandMarksDetector )
    client.create_backend( "face_combiner", fc.combine_faces )
    client.create_backend( "image_downloader", iul.download_image )
    # %%
    client.create_endpoint( "upload_image", backend="image_uploader",
                            route="/upload_image", methods=["POST"] )

    client.create_endpoint( "detect_faces", backend="face_detector",
                            route="/detect_faces" )

    client.create_endpoint( "detect_landmarks", backend="landmark_detector",
                            route="/detect_landmarks", methods=["GET"] )

    client.create_endpoint("combine_faces", backend="face_combiner",
                           route="/combine_faces", methods=["POST"])

    client.create_endpoint("download_image", backend="image_downloader",
                           route="/download_image", methods=["GET"])

    # %%
    while True:
        print( f"Keep alive loop: {dt.datetime.utcnow()}" )
        log.info( f"Keep alive loop: {dt.datetime.utcnow()}" )
        time.sleep( 600 )


def _interactive_testing( ):
    # %%
    # Query our endpoint in two different ways: from HTTP and from Python.
    # %%
    ray.init(address="127.0.0.1:8000")
    client = serve.connect( )
    # %%
    host = "http://127.0.0.1:8000"

    fpath = Path("/home/teo/Dokumente/personal/Photos/teo-2019-01-01.png")
    with open( fpath, "rb" ) as f_in:
        data = f_in.read()

    ext = fpath.suffix[1:]
    # %%
    resp = requests.post( host + "/upload_image",
                          headers={"Content-Type": "image/" + ext },
                          data=data )
    print( resp.status_code, resp.text )
    resp_json = json.loads( resp.text )
    # %%
    img_key = "f797512df23a91d0.png"
    # %%
    resp2 = requests.get( f"{host}/detect_faces", params=dict(img_key=img_key) )

    print( resp2.text )
    # %%
    resp3 = requests.get( f"{host}/detect_landmarks",
                          params=dict(img_key=img_key, face_idx=0))

    print( resp3.text )
    # %%
    # > {"count": 1}
    print(ray.get(client.get_handle("my_endpoint").remote()))
    # > {"count": 2}


if __name__ == "__main__":
    _set_up_endpoints()
