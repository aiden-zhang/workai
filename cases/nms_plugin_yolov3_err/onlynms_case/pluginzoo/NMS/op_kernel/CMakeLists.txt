cmake_minimum_required (VERSION 3.5)

set(PLUGIN_SO_NAME DLNms_opt_plugin)
set(TVM_PLUGIN_SO_NAMEA DLNms_opt_tvm)
# project(${PLUGIN_SO_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

file(GLOB PLUGIN_TVM_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB PLUGIN_CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

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
add_library(${TVM_PLUGIN_SO_NAMEA} SHARED ${PLUGIN_TVM_SOURCES})
set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES LANGUAGE CXX)
add_library (${PLUGIN_SO_NAME} SHARED ${PLUGIN_SOURCES} ${PLUGIN_CU_SOURCES})

# add_library(${TVM_PLUGIN_SO_NAMEA}_static STATIC ${PLUGIN_TVM_SOURCES})
# add_library(${PLUGIN_SO_NAME}_static STATIC  ${PLUGIN_SOURCES} ${PLUGIN_CU_SOURCES})

set(CMAKE_CXX_COMPILER "$ENV{SDK_DIR}/bin/clang++")

target_include_directories(${TVM_PLUGIN_SO_NAMEA} PUBLIC $ENV{SDK_DIR}/include)
target_include_directories(${PLUGIN_SO_NAME} PUBLIC $ENV{SDK_DIR}/include)

target_compile_options (${PLUGIN_SO_NAME}        PRIVATE -DNDEBUG  -fPIC --cuda-gpu-arch=dlgpuc64 -x cuda -fstack-protector)
# target_compile_options (${PLUGIN_SO_NAME}_static PRIVATE -DNDEBUG  -fPIC --cuda-gpu-arch=dlgpuc64 -x cuda -fstack-protector)

target_link_libraries(${TVM_PLUGIN_SO_NAMEA}
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
                      ${DL_SDK_LIB_DIR}/libdlhc.so
                      ${DL_SDK_LIB_DIR}/libcurt.so
                      ${DL_SDK_LIB_DIR}/libdlhal.so
                      ${DL_SDK_LIB_DIR}/libdli_tu.so
                      ${DL_SDK_LIB_DIR}/libclang-cpp.so
                      ${DL_SDK_LIB_DIR}/libclang.so)



# set_target_properties(${PLUGIN_SO_NAME}
#   PROPERTIES LIBRARY_OUTPUT_NAME ${PLUGIN_SO_NAME}
#   PREFIX ""
#   DEBUG_POSTFIX ""
#   LINK_FLAGS "-Wl,-z,defs")

INSTALL (TARGETS ${PLUGIN_SO_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
INSTALL (TARGETS ${TVM_PLUGIN_SO_NAMEA} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# INSTALL (TARGETS ${PLUGIN_SO_NAME}_static LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# INSTALL (TARGETS ${TVM_PLUGIN_SO_NAMEA}_static LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})