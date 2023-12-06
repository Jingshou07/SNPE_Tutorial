#!/bin/bash

rm -r build
mkdir -p build
rm -r output
mkdir -p output
cd build
cmake ../
make -j6

pushd sample-alg
checkinstall -D -y --install=no  --maintainer=THIRDPARTY \
  --pkgversion=1.0.0 --pkgrelease=3rd --pkggroup=algorithm \
  --pkgname=Alg-Yolov8s-3rd-Linux
popd

cp sample/test-image ../output
cp sample-alg/alg-yolov8s-3rd-linux_1.0.0-3rd_arm64.deb ../output
