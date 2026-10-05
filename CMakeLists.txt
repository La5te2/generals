cmake_minimum_required(VERSION 3.20)

# use existing packages from vcpkg when its location is provided.
if(NOT CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT} AND NOT "$ENV{VCPKG_ROOT}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" vcpkgRoot)
    set(CMAKE_TOOLCHAIN_FILE "${vcpkgRoot}/scripts/buildsystems/vcpkg.cmake"
        CACHE FILEPATH "CMake toolchain file")
endif()
option(VCPKG_MANIFEST_INSTALL "Install packages during configuration" OFF)

project(Nebula LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# engine is header-only. consumers inherit its include path and C++20 requirement.
add_library(engine INTERFACE)
target_include_directories(engine INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(engine INTERFACE cxx_std_20)

add_library(agents STATIC src/agents/builtin.cpp)
target_link_libraries(agents PUBLIC engine)

add_library(match STATIC src/interface/match.cpp)
target_link_libraries(match PUBLIC agents)

option(NEBULA_BUILD_INTERFACE "Build the interface application" ON)
if(NEBULA_BUILD_INTERFACE)
    find_package(glad CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)

    add_executable(interface src/interface/nebula.cpp src/interface/renderer.cpp src/interface/console.cpp)
    target_link_libraries(interface PRIVATE match glad::glad glfw)
    if(WIN32)
        target_link_libraries(interface PRIVATE user32)
    endif()
    set_target_properties(interface PROPERTIES OUTPUT_NAME nebula)
    add_custom_command(TARGET interface POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/ATTRIBUTION.txt"
            "$<TARGET_FILE_DIR:interface>/ATTRIBUTION.txt"
        VERBATIM)
endif()
