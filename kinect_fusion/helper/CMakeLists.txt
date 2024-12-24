 include(FetchContent)

set(BUILD_TESTING OFF)
 set(EIGEN_BUILD_TESTING OFF)
 set(EIGEN_MPL2_ONLY ON)
 set(EIGEN_BUILD_PKGCONFIG OFF)
 set(EIGEN_BUILD_DOC OFF)

 FetchContent_Declare(
     Eigen
     GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
     GIT_TAG        3.4.0
     GIT_SHALLOW    TRUE
 )

 FetchContent_GetProperties(Eigen)
 if(NOT Eigen_POPULATED)
     FetchContent_Populate(Eigen)
     add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
 endif()


set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../Libs)
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    endif()
endif(WIN32)

set(Flann_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/flann/)

find_package(Eigen3 REQUIRED)

add_library(helper STATIC FreeImageHelper.cpp Eigen.h)
target_link_libraries(helper PUBLIC  Eigen3::Eigen freeimage)
target_include_directories(helper PUBLIC . ${FreeImage_INCLUDE_DIR} ${Flann_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})

if(WIN32)
    set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
    set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")

    target_include_directories(helper PUBLIC ${FreeImage_INCLUDE_DIR})
    target_link_directories(helper PUBLIC ${FreeImage_LIBRARY_DIR})

     add_custom_command(TARGET helper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FreeImage_LIBRARY_DIR}/FreeImage.dll"
            ${CMAKE_BINARY_DIR})
endif()
