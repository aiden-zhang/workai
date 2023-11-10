export SDK_DIR=${DLICC_PATH}/../

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 
echo $DIR

rm build -rf
rm -rf pluginzoo/dlnne_plugin_build
rm sample/model/*.engine
# exit 0

cd pluginzoo
./run_me.sh

cd ${DIR}/

build_path=${DIR}/build

if [ ! -d "$build_path" ]; then
mkdir ${DIR}/build
fi

mkdir ${DIR}/build
cd ${DIR}/build
cmake ${DIR}/sample/c++/
make 
cd bin

# exit 0
echo "------------------ the first run-----------------"
./yolov3 ../../sample/model/yolov5s_nms.onnx ../../sample/imageonlyone/  1


echo "------------------the second run-----------------"
./yolov3 ../../sample/model/yolov5s_nms.onnx ../../sample/imageonlyone/  1

