#!/usr/bin/env bash
export SDK_DIR=${DLICC_PATH}/../


if [ -z $SHELL_FOLDER ]; then
    SHELL_FOLDER=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)
fi

rm -rf ${SHELL_FOLDER}/dlnne_plugin_build

echo "SHELL_FOLDER:$SHELL_FOLDER"
build_path=$SHELL_FOLDER/dlnne_plugin_build

if [ ! -d "$build_path" ]; then
    mkdir $SHELL_FOLDER/dlnne_plugin_build
fi

echo "sdk_dir:${SDK_DIR}"
cd $SHELL_FOLDER/dlnne_plugin_build
if [ ! -f "/mercury/utility/bin/cmake" ]; then
    cmake ../dlnne_plugin/
else
    /mercury/utility/bin/cmake ../dlnne_plugin/
fi

# 1. build library
make -j4
cd $SHELL_FOLDER


# 2. run test
python3 test_plugin_op.py



