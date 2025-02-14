# Get the name of the current directory
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(SOURCES texture_3d.cpp)

# Find all shader files
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_SLANG ${SHD_DIR}/perlin.slang ${SHD_DIR}/raster.slang ${SHD_DIR}/icp.slang)
set(SHADER_HEADERS ${SHD_DIR}/device_host.h)

# Create the sample
create_sample(${PROJECT_NAME} 
  SOURCES           ${SOURCES}
  SHADER_HEADERS    ${SHADER_HEADERS}
  SHADER_SLANG      ${SHADER_SLANG}
)

add_subdirectory(helper)
target_link_libraries("vk_mini_${PROJECT_NAME}" PUBLIC helper)

find_package(OpenMP)

create_sample(fusion_exe
    SOURCES fusion_main.cpp
    SHADER_HEADERS ${SHD_DIR}/device_host.h
    SHADER_SLANG ${SHD_DIR}/perlin.slang
)
target_link_libraries(vk_mini_fusion_exe PUBLIC helper)

if(CUDAToolkit_FOUND)
    set(_cuda_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/cuda_gen")
    file(MAKE_DIRECTORY ${_cuda_gen_dir})

    add_custom_command(
        OUTPUT "${_cuda_gen_dir}/fusion_cuda.cu"
        COMMAND
            ${Slang_slangc_EXE}
            -target cuda
            -o "${_cuda_gen_dir}/tmp_fusion_cuda.cu"
            -I${SHD_DIR} -I${SAMPLES_ROOT_DIR} -I${NVPRO_CORE_DIR}
            ${SHD_DIR}/perlin.slang
        COMMAND
            ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/fix_generated_cuda.py
            "${_cuda_gen_dir}/tmp_fusion_cuda.cu"
        DEPENDS
            ${SHD_DIR}/perlin.slang
            ${CMAKE_CURRENT_SOURCE_DIR}/fix_generated_cuda.py
    )
    add_library(compiled_cuda INTERFACE "${_cuda_gen_dir}/fusion_cuda.cu")

    add_library(cuda_fusion STATIC fusion.cu)
    target_link_libraries(cuda_fusion PUBLIC CUDA::cudart helper compiled_cuda)
    target_compile_definitions(cuda_fusion PUBLIC CUDA_AVAILABLE)
    target_compile_options(cuda_fusion PRIVATE -diag-suppress=20013 -diag-suppress=20015)
    target_link_libraries(vk_mini_fusion_exe PUBLIC cuda_fusion)
    target_include_directories(cuda_fusion PRIVATE ${_cuda_gen_dir})
endif()

create_sample(viewer
        SOURCES viewer.cpp
        SHADER_HEADERS ${SHD_DIR}/device_host.h
        SHADER_SLANG ${SHD_DIR}/raster.slang
)
target_link_libraries(vk_mini_viewer PUBLIC helper)

if (OpenMP_FOUND)
    target_link_libraries("vk_mini_${PROJECT_NAME}" PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(vk_mini_fusion_exe PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(vk_mini_viewer PUBLIC OpenMP::OpenMP_CXX)
endif()
