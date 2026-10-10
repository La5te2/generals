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

option(NEBULA_BUILD_AGENT "Build the Nebula LibTorch inference program" ON)
option(SIMPLE_BUILD_AGENT "Build the Simple GenFormer LibTorch inference program" ON)
set(inferenceAgents)
if(NEBULA_BUILD_AGENT)
    list(APPEND inferenceAgents nebula)
endif()
if(SIMPLE_BUILD_AGENT)
    list(APPEND inferenceAgents simple)
endif()
if(inferenceAgents)
    find_package(Torch CONFIG QUIET)
    if(NOT Torch_FOUND)
        # PyTorch exposes the CMake package location for its bundled C++ libraries.
        if(WIN32)
            find_program(Python3_EXECUTABLE NAMES python python3)
        else()
            find_program(Python3_EXECUTABLE NAMES python3 python)
        endif()
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            execute_process(
                COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
                RESULT_VARIABLE torchStatus
                OUTPUT_VARIABLE torchPrefix
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                TIMEOUT 20)
            if(torchStatus EQUAL 0 AND NOT "${torchPrefix}" STREQUAL "")
                list(APPEND CMAKE_PREFIX_PATH "${torchPrefix}")
            endif()
        endif()
        find_package(Torch CONFIG REQUIRED)
    endif()
    separate_arguments(torchFlags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    if(WIN32)
        file(GLOB torchDlls "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    endif()
    foreach(agent IN LISTS inferenceAgents)
        add_executable(${agent} "src/agents/${agent}/infer.cpp")
        target_link_libraries(${agent} PRIVATE engine "${TORCH_LIBRARIES}")
        target_compile_options(${agent} PRIVATE ${torchFlags})
        set_target_properties(${agent} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${agent}"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${agent}"
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${agent}")
        if(WIN32 AND torchDlls)
            add_custom_command(TARGET ${agent} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${torchDlls} "$<TARGET_FILE_DIR:${agent}>"
                VERBATIM)
        endif()
    endforeach()
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
    add_library(lan STATIC src/interface/lan.cpp)
    target_link_libraries(lan PUBLIC local PRIVATE LibDataChannel::LibDataChannel nlohmann_json::nlohmann_json)
    find_package(glad CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)

    add_executable(interface WIN32 src/interface/interface.cpp src/interface/renderer.cpp src/interface/human.cpp
        src/interface/console.cpp src/interface/dialog.cpp)
    target_link_libraries(interface PRIVATE local online lan glad::glad glfw)
    if(WIN32)
        target_link_libraries(interface PRIVATE user32 comdlg32 ole32 shell32 uuid)
    endif()
    set_target_properties(interface process local online lan PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/interface"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/interface"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/interface")
    add_custom_command(TARGET interface POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/ATTRIBUTION.txt"
            "$<TARGET_FILE_DIR:interface>/ATTRIBUTION.txt"
        VERBATIM)
endif()
