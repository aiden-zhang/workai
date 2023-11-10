#!/usr/bin/env bash
CURRENT_PATH=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)
export SDK_DIR=${DLICC_PATH}/../

if [ -z $SHELL_FOLDER ]; then
SHELL_FOLDER=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)
fi

echo $SHELL_FOLDER
build_path=$SHELL_FOLDER/build

if [ ! -d "$build_path" ]; then
mkdir $SHELL_FOLDER/build
fi

cd $SHELL_FOLDER/build
if [ ! -f "/mercury/utility/bin/cmake" ]; then
    cmake ..
else
    /mercury/utility/bin/cmake ..
fi
make -j4
make install
cd $SHELL_FOLDER/lib
# dlcc -shared -o libdlAll_plugin.so -L. -Wl,--whole-archive libSpatialTransformer_opt_plugin_static.a libDLNms_opt_plugin_static.a libupsample_opt_plugin_static.a libPriorBox_opt_plugin_static.a libNormalize_opt_plugin_static.a  -Wl,--no-whole-archive
cd $SHELL_FOLDER/../
cd $CURRENT_PATH
rm -rf build
