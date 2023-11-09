#!/bin/bash
# CASE_ROOT_PATH=$(cd $(dirname $0); pwd)
# echo "CASE ROOT PATH=${CASE_ROOT_PATH}"

case_name=perf_test

echo "<customer_bugs_test_case::${case_name} RUN>"

# cd ${CASE_ROOT_PATH}

# rm -rf ./build
# mkdir build
# cd build
# cmake  ../
# make
make -j
# ./main -m /work/for_release/pb_model/resnet_v1_50_quant_tu_nchw.pb -b 32 -e 1 -s 1 -d 0 -t 6 -o resnet_v1_50/SpatialSqueeze
./a.out -m ./justadduint8.rlym -b8 -s 1 -d 0  #saved to ./slz.bin
echo -e  "--------------------------------------------------------------------\n\n"
./a.out -b8 -e 1000 -s 0 -d 1 -t 1  #load from ./slz.bin
if [[ $? -eq 0 ]];then
    echo "<customer_bugs_test_case::${case_name} PASS>"
else
    echo "<customer_bugs_test_case::${case_name} FAIL>"
fi

#rm -rf ${CASE_ROOT_PATH}/build/
