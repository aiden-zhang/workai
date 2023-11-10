aux_source_directory(. DIR_LIB_SRCS)
SET(CMAKE_C_COMPILER "${SDK_DIR}/bin/dlcc")
SET(CMAKE_CXX_COMPILER "${SDK_DIR}/bin/dlcc")
set(SDK_DIR $ENV{DLICC_PATH}/../)
INCLUDE_DIRECTORIES(${SDK_DIR}/include)
link_directories(${SDK_DIR}/lib)


SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
add_library(Mydljpu SHARED ${DIR_LIB_SRCS})
target_compile_options(Mydljpu PRIVATE
                       -std=c++11
                       -Wno-c++11-narrowing -DNDEBUG
                       -Wdouble-promotion -fPIC --cuda-gpu-arch=dlgpuc64
                       -x cuda)