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

echo "\n----saved engine to ./slz.bin----------------"
./a.out -m ./justadduint8.rlym  -s1

echo -e  "\n\n----load engine from ./slz.bin-----------------"
./a.out -d1  -t1




if [[ $? -eq 0 ]];then
    echo "<customer_bugs_test_case::${case_name} PASS>"
else
    echo "<customer_bugs_test_case::${case_name} FAIL>"
fi

#rm -rf ${CASE_ROOT_PATH}/build/
