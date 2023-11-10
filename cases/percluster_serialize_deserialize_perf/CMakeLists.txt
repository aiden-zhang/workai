cmake_minimum_required(VERSION 3.5)
project(main)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cc)

set(SDK_DIR $ENV{DLICC_PATH}/../)

include_directories(
${SDK_DIR}/include
${SDK_DIR}/include/dlnne
)

link_directories(${SDK_DIR}/lib)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE curt dlnne pthread)