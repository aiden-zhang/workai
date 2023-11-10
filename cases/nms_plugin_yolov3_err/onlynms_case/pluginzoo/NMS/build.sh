#!/usr/bin/env bash
CURRENT_PATH=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)

if [ -z $SHELL_FOLDER ]; then
SHELL_FOLDER=$(cd $(dirname "${BASH_SOURCE[0]}");pwd)
fi

if [ -z "${SDK_DIR}" ] || [ ! -d ${SDK_DIR} ];then
    export SDK_DIR=$SHELL_FOLDER/../../../
else

    export SDK_DIR=${SDK_DIR}
fi

echo $SHELL_FOLDER
build_path=$SHELL_FOLDER/dlnne_plugin_build

if [ ! -d "$build_path" ]; then
mkdir $SHELL_FOLDER/dlnne_plugin_build
fi

echo "sdk_dir:${SDK_DIR}"
cd $SHELL_FOLDER/dlnne_plugin_build
if [ ! -f "/mercury/utility/bin/cmake" ]; then
    cmake ../op_kernel/
else
    /mercury/utility/bin/cmake ../op_kernel/
fi
make -j4
cd $SHELL_FOLDER/../

cd $CURRENT_PATH
