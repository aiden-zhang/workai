#aux_source_directory(. DIR_LIB_SRCS)
SET(CMAKE_C_COMPILER "${SDK_DIR}/bin/dlcc")
SET(CMAKE_CXX_COMPILER "${SDK_DIR}/bin/dlcc")
INCLUDE_DIRECTORIES($ENV{SDK_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/imageProcess/include)

set(CMAKE_BUILD_TYPE "Debug")

file(GLOB_RECURSE DIR_LIB_SRCS src/*.cc src/*.cu)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
add_library(MyimgPre SHARED ${DIR_LIB_SRCS})
target_compile_options(MyimgPre PRIVATE
                       -std=c++11
                       -Wno-c++11-narrowing -DNDEBUG
                       -Wdouble-promotion -fPIC --cuda-gpu-arch=dlgpuc64
                       -x cuda)
