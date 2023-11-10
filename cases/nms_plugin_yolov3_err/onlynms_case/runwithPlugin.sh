#!/bin/bash
export SDK_DIR=${DLICC_PATH}/../

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 
echo $DIR

rm pluginzoo/lib  pluginzoo/include ./a.out   -rf 

## 1. build to produce so
echo  -e "\n\n----------- now build library ------------"
cd pluginzoo/
./build_all.sh
# exit 0

cd ${DIR}/

## 2. build a.out
echo -e "\n\n----------- now build a.out ------------"
dlcc nmstest.cc -Wl,-rpath=${DIR}/pluginzoo/lib -L${DIR}/pluginzoo/lib -ldlnne  -lcurt -lDLNms_opt_plugin -I${SDK_DIR}/include/dlnne

## 3. run 
exit 0
echo  -e "\n\n----------- now run a.out ------------"
./a.out ./onlynmsplugin_discard_filtersort.onnx  1
