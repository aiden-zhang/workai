project(dl_plugin_utils)
cmake_minimum_required(VERSION 3.1)


set(SDK_PATH $ENV{DLICC_PATH}/../)
file(GLOB src_file *.cc)
add_library(dlplugin_utils ${src_file})

set(CMAKE_CXX_COMPILER dlcc)
target_compile_options(dlplugin_utils PRIVATE
                       -std=c++11
                       -Wno-c++11-narrowing -DNDEBUG
                       -Wdouble-promotion -fPIC --cuda-gpu-arch=dlgpuc64
                       -x cuda)
include_directories(dlplugin_utils PUBLIC ${SDK_PATH}/include/dlnne)

INSTALL (TARGETS dlplugin_utils DESTINATION ${CMAKE_INSTALL_LIBDIR})
