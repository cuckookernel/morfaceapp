# morfaceapp

Developed on Android-studio. Start with  `~/bin/android-studio/bin/studio.sh`

Install Python environment:

```bash
mkdir -p ./venv/    # this dir is in gitignore
python3.10 -m venv ./venv/py310-torch-fld
source ./venv/py310-torch-fld/bin/activate.sh
pip install -r requirements.py310.txt
```


## To run gradio demos:

```bash
source ./venv/py310-torch-fld/bin/activate.sh
```


### Just face detection

```
python gradio/face_detect_v2.py
```

### Face detection & Landmark detection

```
python gradio/face_lm_detect.py
```

### Landmark Detection on two face + morphing

```
python gradio/morface.py
```
