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

add_executable(simple src/agents/simple.cpp)
target_link_libraries(simple PRIVATE engine)

add_library(arena SHARED src/agents/nebula/arena.cpp)
target_link_libraries(arena PRIVATE engine)
set_target_properties(arena PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nebula"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nebula"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nebula")

option(NEBULA_BUILD_AGENT "Build the LibTorch inference program" OFF)
if(NEBULA_BUILD_AGENT)
    find_package(Torch CONFIG REQUIRED)
    add_executable(nebula src/agents/nebula/infer.cpp)
    target_link_libraries(nebula PRIVATE engine "${TORCH_LIBRARIES}")
    set_target_properties(nebula PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nebula"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nebula"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/nebula")
    separate_arguments(torchFlags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(nebula PRIVATE ${torchFlags})
    if(WIN32)
        file(GLOB torchDlls "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        if(torchDlls)
            add_custom_command(TARGET nebula POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${torchDlls} "$<TARGET_FILE_DIR:nebula>"
                VERBATIM)
        endif()
    endif()
endif()

option(NEBULA_BUILD_INTERFACE "Build the interface application" ON)
if(NEBULA_BUILD_INTERFACE)
    find_package(Threads REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)
    add_library(process STATIC src/interface/process.cpp)
    target_link_libraries(process PUBLIC engine Threads::Threads)
    add_library(local STATIC src/interface/local.cpp src/interface/replay.cpp)
    target_link_libraries(local PUBLIC process PRIVATE nlohmann_json::nlohmann_json)
    find_package(LibDataChannel CONFIG REQUIRED)
    add_library(online STATIC src/interface/online.cpp)
    target_link_libraries(online PUBLIC process PRIVATE LibDataChannel::LibDataChannel nlohmann_json::nlohmann_json)
    find_package(glad CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)

    add_executable(interface WIN32 src/interface/interface.cpp src/interface/renderer.cpp src/interface/controller.cpp
        src/interface/console.cpp src/interface/dialog.cpp)
    target_link_libraries(interface PRIVATE local online glad::glad glfw)
    if(WIN32)
        target_link_libraries(interface PRIVATE user32 comdlg32 ole32 shell32 uuid)
    endif()
    set_target_properties(interface process local online PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/interface"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/interface"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/interface")
    add_custom_command(TARGET interface POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/ATTRIBUTION.txt"
            "$<TARGET_FILE_DIR:interface>/ATTRIBUTION.txt"
        VERBATIM)
endif()
