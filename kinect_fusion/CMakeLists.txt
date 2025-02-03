# Get the name of the current directory
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(SOURCES texture_3d.cpp)

# Find all shader files
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_SLANG ${SHD_DIR}/perlin.slang ${SHD_DIR}/raster.slang)
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

if (OpenMP_FOUND)
    target_link_libraries("vk_mini_${PROJECT_NAME}" PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(vk_mini_fusion_exe PUBLIC OpenMP::OpenMP_CXX)
endif()
