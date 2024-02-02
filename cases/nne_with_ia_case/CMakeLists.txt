cmake_minimum_required(VERSION 3.2.0)


set(CMAKE_BUILD_TYPE "Debug")
project(imageProcess)
set(CMAKE_CXX_COMPILER "dlcc")
if ("$ENV{DLI_V2}" STREQUAL "ON")
    set(gpu_arch --cuda-gpu-arch=dlgput64)
    set(src_path src_gen2)
    set(file_suffix ".cu.cc")
    message(STATUS "***v2 sdk***")
else()
    set(gpu_arch --cuda-gpu-arch=dlgpuc64)
    set(src_path src_gen1)
    set(file_suffix "*")
    message(STATUS "***v1 sdk***")
endif()

set(SDK_DIR $ENV{DLICC_PATH}/../)
include_directories(${SDK_DIR}/include)
include_directories(${SDK_DIR}/include/dlnne)
include_directories(${PROJECT_SOURCE_DIR}/)
include_directories(${PROJECT_SOURCE_DIR}/include)

# # RGB resize
# file(GLOB_RECURSE src_files ${src_path}/rgb_resize.cu.cc test/test_rgb_resize${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_rgb_resize ${src_files})
# target_compile_options(test_rgb_resize PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_rgb_resize PRIVATE curt)

# Gray resize
file(GLOB_RECURSE src_files
    test/test_nne_with_ia.cc 
    test/nne_util.cc
    test/case_register.cc  
    test/util.cc
    ${src_path}/gray_resize.cu.cc)

add_executable(test_gray_resize ${src_files})
target_compile_options(test_gray_resize PRIVATE
                       -Wno-c++11-narrowing -DNDEBUG -std=c++11
                       -Wdouble-promotion -fPIC ${gpu_arch}
                       -x cuda -g)
                  
target_link_libraries(test_gray_resize PRIVATE curt dlnne pthread)

# # YUV resize
# file(GLOB_RECURSE src_files ${src_path}/yuv_resize.cu.cc test/test_yuv_resize${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_yuv_resize ${src_files})
# target_compile_options(test_yuv_resize PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_yuv_resize PRIVATE curt)

# # YUV2RGB
# file(GLOB_RECURSE src_files ${src_path}/yuv_to_rgb.cu.cc test/test_yuv_to_rgb${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_yuv_to_rgb ${src_files})
# target_compile_options(test_yuv_to_rgb PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_yuv_to_rgb PRIVATE curt)

# # RGB roi resize
# file(GLOB_RECURSE src_files ${src_path}/rgb_resize.cu.cc test/test_rgb_roi_resize${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_rgb_roi_resize ${src_files})
# target_compile_options(test_rgb_roi_resize PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_rgb_roi_resize PRIVATE curt)

# # RGB resize plane pad
# file(GLOB_RECURSE src_files ${src_path}/rgb_resize.cu.cc test/test_plane_rgb_resize_pad${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_plane_rgb_resize_pad ${src_files})
# target_compile_options(test_plane_rgb_resize_pad PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_plane_rgb_resize_pad PRIVATE curt)

# # RGB resize plane pad norm
# file(GLOB_RECURSE src_files ${src_path}/rgb_resize.cu.cc ${src_path}/rgb_normalization.cu.cc test/test_bilinear_resize_plane_pad_normalization_fuse${file_suffix} test/case_register.cc  test/util.cc)
# add_executable(test_rgb_resize_plane_pad_norm_fuse ${src_files})
# target_compile_options(test_rgb_resize_plane_pad_norm_fuse PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_rgb_resize_plane_pad_norm_fuse PRIVATE curt)

# # YUV2RGB resize plane pad norm
# file(GLOB_RECURSE src_files src_files ${src_path}/yuv2rgb_resize_norm.cu.cc test/test_yuv2rgb_resize_plane_pad_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_yuv2rgb_resize_plane_pad_norm ${src_files})
# target_compile_options(test_yuv2rgb_resize_plane_pad_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_yuv2rgb_resize_plane_pad_norm PRIVATE curt)

# # RGB crop
# file(GLOB_RECURSE src_files src_files ${src_path}/rgb_crop.cu.cc test/test_rgb_crop${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_rgb_crop ${src_files})
# target_compile_options(test_rgb_crop PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_rgb_crop PRIVATE curt)

# # YUV crop
# file(GLOB_RECURSE src_files src_files ${src_path}/yuv_crop.cu.cc test/test_yuv_crop${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_yuv_crop ${src_files})
# target_compile_options(test_yuv_crop PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_yuv_crop PRIVATE curt)

# # YUV2BGR resize plane pad norm
# file(GLOB_RECURSE src_files src_files ${src_path}/yuv2rgb_resize_norm.cu.cc test/test_yuv2bgr_resize_plane_pad_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_yuv2bgr_resize_plane_pad_norm ${src_files})
# target_compile_options(test_yuv2bgr_resize_plane_pad_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_yuv2bgr_resize_plane_pad_norm PRIVATE curt)

# # RGB roi resize norm
# file(GLOB_RECURSE src_files ${src_path}/rgb_roi_resize_norm.cu.cc test/test_fuse_rgb_roi_resize_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_rgb_roi_resize_norm ${src_files})
# target_compile_options(test_rgb_roi_resize_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_rgb_roi_resize_norm PRIVATE curt)

# # roi yuv2rgb resize
# file(GLOB_RECURSE src_files ${src_path}/yuv2rgb_resize_norm.cu.cc test/test_roi_yuv2rgb_resize_plane_pad.cu.cc test/case_register.cc test/util.cc)
# add_executable(test_roi_yuv2rgb_resize ${src_files})
# target_compile_options(test_roi_yuv2rgb_resize PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_roi_yuv2rgb_resize PRIVATE curt)

# # roi yuv2rgb resize norm
# file(GLOB_RECURSE src_files ${src_path}/yuv2rgb_resize_norm.cu.cc test/test_roi_yuv2rgb_resize_plane_pad_norm.cu.cc test/case_register.cc test/util.cc)
# add_executable(test_roi_yuv2rgb_resize_norm ${src_files})
# target_compile_options(test_roi_yuv2rgb_resize_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_roi_yuv2rgb_resize_norm PRIVATE curt)

# # full range YUV2RGB resize plane pad norm
# file(GLOB_RECURSE src_files src_files ${src_path}/yuv2rgb_resize_norm.cu.cc test/test_full_range_yuv2rgb_resize_plane_pad_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_full_range_yuv2rgb_resize_plane_pad_norm ${src_files})
# target_compile_options(test_full_range_yuv2rgb_resize_plane_pad_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_full_range_yuv2rgb_resize_plane_pad_norm PRIVATE curt)

# # RGB roi affine
# file(GLOB_RECURSE src_files ${src_path}/rgb_affine.cu.cc test/test_rgb_roi_affine${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_rgb_roi_affine ${src_files})
# target_compile_options(test_rgb_roi_affine PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_rgb_roi_affine PRIVATE curt)

# # yuv2rgb affine
# file(GLOB_RECURSE src_files ${src_path}/yuv2rgb_affine.cu.cc test/test_yu122rgb_affine${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_yuv2rgb_affine ${src_files})
# target_compile_options(test_yuv2rgb_affine PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_yuv2rgb_affine PRIVATE curt)

# # roi yuv444p2rgb resize norm
# file(GLOB_RECURSE src_files ${src_path}/yuv2rgb_resize_norm.cu.cc test/test_roi_yuv444p2rgb_resize_plane_pad_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_roi_yuv444p2rgb_resize_norm ${src_files})
# target_compile_options(test_roi_yuv444p2rgb_resize_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_roi_yuv444p2rgb_resize_norm PRIVATE curt)

# # roi yuv400p2rgb resize norm
# file(GLOB_RECURSE src_files ${src_path}/yuv2rgb_resize_norm.cu.cc test/test_roi_yuv400p2rgb_resize_plane_pad_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_roi_yuv400p2rgb_resize_norm ${src_files})
# target_compile_options(test_roi_yuv400p2rgb_resize_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_roi_yuv400p2rgb_resize_norm PRIVATE curt)

# # roi yuv422p2rgb resize norm
# file(GLOB_RECURSE src_files ${src_path}/yuv2rgb_resize_norm.cu.cc test/test_roi_yuv422p2rgb_resize_plane_pad_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_roi_yuv422p2rgb_resize_norm ${src_files})
# target_compile_options(test_roi_yuv422p2rgb_resize_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_roi_yuv422p2rgb_resize_norm PRIVATE curt)

# # roi yuv4222rgb resize norm
# file(GLOB_RECURSE src_files ${src_path}/yuv2rgb_resize_norm.cu.cc test/test_roi_yuv4222rgb_resize_plane_pad_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_roi_yuv4222rgb_resize_norm ${src_files})
# target_compile_options(test_roi_yuv4222rgb_resize_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_roi_yuv4222rgb_resize_norm PRIVATE curt)

# # roi yuv2rgb affine norm
# file(GLOB_RECURSE src_files ${src_path}/yuv2rgb_affine.cu.cc test/test_roi_yuv2rgb_affine_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_roi_yuv2rgb_affine_norm ${src_files})
# target_compile_options(test_roi_yuv2rgb_affine_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_roi_yuv2rgb_affine_norm PRIVATE curt)

# # roi yuv2rgb resize
# file(GLOB_RECURSE src_files ${src_path}/yuv2rgb_resize_norm.cu.cc test/test_roi_yuv2rgb_resize_quantize_plane_pad${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_roi_yuv2rgb_resize_quantize ${src_files})
# target_compile_options(test_roi_yuv2rgb_resize_quantize PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_roi_yuv2rgb_resize_quantize PRIVATE curt)

# # roi yuv2rgb pad resize norm torch
# file(GLOB_RECURSE src_files ${src_path}/pytorch_resize_norm.cu.cc test/test_roi_yuv2rgb_pad_resize_plane_norm_torch${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_roi_yuv2rgb_pad_resize_norm_torch ${src_files})
# target_compile_options(test_roi_yuv2rgb_pad_resize_norm_torch PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_roi_yuv2rgb_pad_resize_norm_torch PRIVATE curt)

# # radix sort
# file(GLOB_RECURSE src_files ${src_path}/radix_sort.cu.cc test/test_radix_sort${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_radix_sort ${src_files})
# target_compile_options(test_radix_sort PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++14
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_radix_sort PRIVATE curt)

# # YUV2RGB
# file(GLOB_RECURSE src_files ${src_path}/rgb2yuv.cu.cc test/test_rgb2yu12${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_rgb2yuv ${src_files})
# target_compile_options(test_rgb2yuv PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_rgb2yuv PRIVATE curt)

# # nms
# file(GLOB_RECURSE src_files ${src_path}/non_max_suppression.cu.cc test/test_non_max_suppression${file_suffix} test/util.cc)
# add_executable(test_non_max_suppression ${src_files})
# target_compile_options(test_non_max_suppression PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_non_max_suppression PRIVATE curt)

# # yuv_cvt
# file(GLOB_RECURSE src_files ${src_path}/yuv2yuv_cvt.cu.cc test/test_yuv_to_yuv420p${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_yuv_to_yuv420p ${src_files})
# target_compile_options(test_yuv_to_yuv420p PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_yuv_to_yuv420p PRIVATE curt)

# # rgb resize crop norm torch
# file(GLOB_RECURSE src_files ${src_path}/pytorch_resize_norm.cu.cc test/test_torch_rgb_resize_crop_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_torch_rgb_resize_crop_norm ${src_files})
# target_compile_options(test_torch_rgb_resize_crop_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_torch_rgb_resize_crop_norm PRIVATE curt)

# # RGB rotate
# file(GLOB_RECURSE src_files ${src_path}/rgb_rotate.cu.cc test/test_rgb_rotate${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_rgb_rotate ${src_files})
# target_compile_options(test_rgb_rotate PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_rgb_rotate PRIVATE curt)


# # batch RGB roi affine
# file(GLOB_RECURSE src_files ${src_path}/rgb_affine.cu.cc test/test_batch_rgb_roi_affine${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_batch_rgb_roi_affine ${src_files})
# target_compile_options(test_batch_rgb_roi_affine PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_batch_rgb_roi_affine PRIVATE curt)

# # batch RGB resize norm
# file(GLOB_RECURSE src_files ${src_path}/rgb_roi_resize_norm.cu.cc test/test_batch_rgb_resize_norm${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_batch_rgb_resize_norm ${src_files})
# target_compile_options(test_batch_rgb_resize_norm PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_batch_rgb_resize_norm PRIVATE curt)

# # multi roi RGB resize norm
# file(GLOB_RECURSE src_files ${src_path}/rgb_roi_resize_norm.cu.cc test/test_multi_roi_rgb_resize_norm_pad_plane${file_suffix} test/case_register.cc test/util.cc)
# add_executable(test_multi_roi_rgb_resize_norm_pad_plane ${src_files})
# target_compile_options(test_multi_roi_rgb_resize_norm_pad_plane PRIVATE
#                        -Wno-c++11-narrowing -DNDEBUG -std=c++11
#                        -Wdouble-promotion -fPIC ${gpu_arch}
#                        -x cuda)
# target_link_libraries(test_multi_roi_rgb_resize_norm_pad_plane PRIVATE curt)

# execute_process(COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_rgb_resize.sh ./
#                 COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_gray_resize.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_yuv_resize.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_yuv_to_rgb.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_rgb_resize_plane_pad_norm_fuse.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_yuv2rgb_resize_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_rgb_roi_resize.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_rgb_crop.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_yuv_crop.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_plane_rgb_resize_pad.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_yuv2bgr_resize_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_rgb_roi_resize_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_roi_yuv2rgb_resize.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_roi_yuv2rgb_resize_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_rgb_roi_affine.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_full_range_yuv2rgb_resize_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_yuv2rgb_affine.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_roi_yuv444p2rgb_resize_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_roi_yuv400p2rgb_resize_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_roi_yuv422p2rgb_resize_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_roi_yuv2rgb_affine_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_roi_yuv2rgb_resize_quantize.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_roi_yuv2rgb_pad_resize_plane_norm_torch.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_radix_sort.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_roi_yuv4222rgb_resize_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_rgb2yu12.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_non_max_suppression.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_yuv_to_yuv420p.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_rgb_rotate.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_batch_rgb_roi_affine.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_batch_rgb_resize_norm.sh ./
#                 # COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_multi_roi_rgb_resize_norm_pad_plane.sh ./
#                 # COMMAND chmod +x run_test_multi_roi_rgb_resize_norm_pad_plane.sh
#                 # COMMAND chmod +x run_test_batch_rgb_resize_norm.sh
#                 # COMMAND chmod +x run_test_yuv_to_yuv420p.sh
#                 # COMMAND chmod +x run_test_rgb2yu12.sh
#                 # COMMAND chmod +x run_test_roi_yuv4222rgb_resize_norm.sh
#                 # COMMAND chmod +x run_test_roi_yuv2rgb_pad_resize_plane_norm_torch.sh
#                 # COMMAND chmod +x run_test_roi_yuv2rgb_resize_quantize.sh
#                 # COMMAND chmod +x run_test_roi_yuv2rgb_affine_norm.sh
#                 # COMMAND chmod +x run_test_roi_yuv422p2rgb_resize_norm.sh
#                 # COMMAND chmod +x run_test_roi_yuv400p2rgb_resize_norm.sh
#                 # COMMAND chmod +x run_test_roi_yuv444p2rgb_resize_norm.sh
#                 # COMMAND chmod +x run_test_yuv2rgb_affine.sh
#                 # COMMAND chmod +x run_test_full_range_yuv2rgb_resize_norm.sh
#                 # COMMAND chmod +x run_test_rgb_roi_affine.sh
#                 # COMMAND chmod +x run_test_roi_yuv2rgb_resize.sh
#                 # COMMAND chmod +x run_test_roi_yuv2rgb_resize_norm.sh
#                 # COMMAND chmod +x run_test_rgb_roi_resize_norm.sh
#                 # COMMAND chmod +x run_test_yuv2bgr_resize_norm.sh
#                 # COMMAND chmod +x run_test_plane_rgb_resize_pad.sh
#                 # COMMAND chmod +x run_test_rgb_crop.sh
#                 # COMMAND chmod +x run_test_yuv_crop.sh
#                 # COMMAND chmod +x run_test_rgb_roi_resize.sh
#                 # COMMAND chmod +x run_test_yuv2rgb_resize_norm.sh
#                 # COMMAND chmod +x run_test_rgb_resize_plane_pad_norm_fuse.sh
#                 # COMMAND chmod +x run_test_rgb_resize.sh
#                 COMMAND chmod +x run_test_gray_resize.sh
#                 COMMAND chmod +x run_test_yuv_resize.sh
#                 COMMAND chmod +x run_test_yuv_to_rgb.sh
#                 COMMAND chmod +x run_test_radix_sort.sh
#                 COMMAND chmod +x run_test_non_max_suppression.sh
#                 COMMAND chmod +x run_test_rgb_rotate.sh
#                 COMMAND chmod +x run_test_batch_rgb_roi_affine.sh)


execute_process(COMMAND cp ${PROJECT_SOURCE_DIR}/test/run_test_gray_resize.sh ./
                COMMAND chmod +x run_test_gray_resize.sh)