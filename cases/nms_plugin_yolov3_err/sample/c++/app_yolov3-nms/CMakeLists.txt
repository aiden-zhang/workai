set(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_C_COMPILER "gcc")
SET(CMAKE_CXX_COMPILER "g++")

option(DL_OPENCV_ENABLE "WITH OPENCV ENABLE" ON)
# if(DL_OPENCV_ENABLE)
#     INCLUDE_DIRECTORIES(/usr/local/include/opencv4)
#     add_definitions(-DDL_OPENCV_ENABLE)
#     MESSAGE(STATUS "Project with OpenCV ENABLED.")
# endif()


INCLUDE_DIRECTORIES(${SDK_DIR}/include)
INCLUDE_DIRECTORIES(${SDK_DIR}/include/dlnne)
link_directories(${SDK_DIR}/lib)

AUX_SOURCE_DIRECTORY(./ DIR_SRC)
file(GLOB DIR_SRC *.cu *.cpp)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_executable(yolov3 ${DIR_SRC})

if(DL_OPENCV_ENABLE)
  target_link_libraries(yolov3 dlnne curt dl MyimgPre Mydljpu dljpeg
                    opencv_video opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio
                    -lpthread -lm -dl
                  #   opencv_dnn
                    )
else()
  target_link_libraries(yolov3 dlnne curt dl MyimgPre Mydljpu dljpeg -lpthread -lm -dl)
endif()

target_compile_options(yolov3 PRIVATE
                       -std=c++11
                       -g
                       -Wfatal-errors
                    #    -ferror-limit=1
                       -Wno-c++11-narrowing -DNDEBUG
                       -Wdouble-promotion -fPIC 
                       )

set_target_properties(yolov3 PROPERTIES
                      LINK_FLAGS "-rdynamic -Wl,--disable-new-dtags" 
                      )


set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

