#!/bin/bash
# Run this from repo root!

mkdir -p ./models
cd ./models
# wget https://github.com/cunjian/pytorch_face_landmark/raw/master/checkpoint/mobilefacenet_model_best.pth.tar
wget -O "mobilenet_224_model_best_gdconv_external.pth.tar" "https://docs.google.com/uc?export=download&id=1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe"

cd ..
