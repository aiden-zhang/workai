cmake_minimum_required (VERSION 3.5)

set(PLUGIN_SO_NAME dlnne_plugin)

project(${PLUGIN_SO_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

file(GLOB PLUGIN_TVM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB PLUGIN_CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

if(NOT ($ENV{SDK_DIR} MATCHES ".*sdk"))
  message(FATAL_ERROR "you have not export sdk environment, please export SDK_DIR={your sdk path}, for example export SDK_DIR=/usr/sdk")
endif()

message("compile nne plugin independent use sdk............................")
set(DL_SDK_LIB_DIR $ENV{SDK_DIR}/lib)
set(DL_SDK_INCLUDE_DIR $ENV{SDK_DIR}/include)
set(DL_SDK_INCLUDE_NNE_DIR $ENV{SDK_DIR}/include/dlnne)
include_directories(${DL_SDK_INCLUDE_DIR}
                    ${DL_SDK_INCLUDE_NNE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    $ENV{SDK_DIR}/include/libclc/generic/include
                    $ENV{SDK_DIR}/include/tvm
                    ${CMAKE_CURRENT_SOURCE_DIR}/plugin)


set(SUB_DIRECTORIES
    plugin
   )

foreach (PLUGIN_DIR ${SUB_DIRECTORIES})
    add_subdirectory(${PLUGIN_DIR})
endforeach(PLUGIN_DIR)


message("DLI_CUDA ON............................")
add_library(dlnne_tvm_plugin SHARED ${PLUGIN_TVM_SOURCES})
set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES LANGUAGE CXX)
add_library (${PLUGIN_SO_NAME} SHARED ${PLUGIN_SOURCES} ${PLUGIN_CU_SOURCES})

set(CMAKE_CXX_COMPILER "$ENV{SDK_DIR}/bin/clang++")

target_include_directories(dlnne_tvm_plugin PUBLIC $ENV{SDK_DIR}/include)
target_include_directories(${PLUGIN_SO_NAME} PUBLIC $ENV{SDK_DIR}/include)

target_compile_options (${PLUGIN_SO_NAME} PRIVATE -DNDEBUG  -fPIC --cuda-gpu-arch=dlgpuc64 -x cuda -fstack-protector)

target_link_libraries(dlnne_tvm_plugin
                      PRIVATE
                      ${DL_SDK_LIB_DIR}/libtvm.so
                      ${DL_SDK_LIB_DIR}/libdlnne.so
                      ${DL_SDK_LIB_DIR}/libdlhc.so
                      ${DL_SDK_LIB_DIR}/libcurt.so
                      ${DL_SDK_LIB_DIR}/libdlhal.so
                      ${DL_SDK_LIB_DIR}/libdli_tu.so
                      ${DL_SDK_LIB_DIR}/libclang-cpp.so
                      ${DL_SDK_LIB_DIR}/libclang.so)

target_link_libraries(${PLUGIN_SO_NAME}
                      PRIVATE
                      ${DL_SDK_LIB_DIR}/libtvm.so
                      ${DL_SDK_LIB_DIR}/libdlnne.so
                      dlnne_tvm_plugin
                      ${DL_SDK_LIB_DIR}/libdlhc.so
                      ${DL_SDK_LIB_DIR}/libcurt.so
                      ${DL_SDK_LIB_DIR}/libdlhal.so
                      ${DL_SDK_LIB_DIR}/libdli_tu.so
                      ${DL_SDK_LIB_DIR}/libclang-cpp.so
                      ${DL_SDK_LIB_DIR}/libclang.so)



set_target_properties(${PLUGIN_SO_NAME}
  PROPERTIES LIBRARY_OUTPUT_NAME ${PLUGIN_SO_NAME}
  PREFIX ""
  DEBUG_POSTFIX ""
  LINK_FLAGS "-Wl,-z,defs")

