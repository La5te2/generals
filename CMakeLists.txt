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
# Static and import libraries are link-time artifacts, not runtime files.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/CMakeFiles/lib")

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# engine is header-only. consumers inherit its include path and C++20 requirement.
add_library(engine INTERFACE)
target_include_directories(engine INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(engine INTERFACE cxx_std_20)

option(NEBULA_BUILD_INTERFACE "Build the interface application" ON)
option(NEBULA_BUILD_SERVER "Build the self-hosted WebSocket room server" ON)
if(NEBULA_BUILD_INTERFACE OR NEBULA_BUILD_SERVER)
    find_package(Threads REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)
    add_library(process STATIC src/interface/process.cpp)
    target_link_libraries(process PUBLIC engine Threads::Threads)
    add_library(local STATIC src/interface/local.cpp src/interface/replay.cpp)
    target_link_libraries(local PUBLIC process PRIVATE nlohmann_json::nlohmann_json)
    find_package(LibDataChannel CONFIG REQUIRED)
endif()
if(NEBULA_BUILD_SERVER)
    add_library(rooms STATIC src/server/server.cpp)
    target_link_libraries(rooms PUBLIC local PRIVATE LibDataChannel::LibDataChannel nlohmann_json::nlohmann_json)
    add_executable(server src/server/main.cpp)
    target_link_libraries(server PRIVATE rooms)
    set_target_properties(server PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/server")
    add_custom_command(TARGET server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/ATTRIBUTION.txt" "$<TARGET_FILE_DIR:server>/ATTRIBUTION.txt"
        VERBATIM)
endif()
if(NEBULA_BUILD_INTERFACE)
    add_library(online STATIC src/interface/online.cpp)
    target_link_libraries(online PUBLIC process PRIVATE LibDataChannel::LibDataChannel nlohmann_json::nlohmann_json)
    add_library(custom STATIC src/interface/custom.cpp)
    target_link_libraries(custom PUBLIC local PRIVATE LibDataChannel::LibDataChannel nlohmann_json::nlohmann_json)
    find_package(glad CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)

    add_executable(interface WIN32 src/interface/interface.cpp src/interface/renderer.cpp src/interface/human.cpp
        src/interface/console.cpp src/interface/dialog.cpp)
    target_link_libraries(interface PRIVATE local online custom glad::glad glfw)
    if(WIN32)
        target_link_libraries(interface PRIVATE user32 comdlg32 ole32 shell32 uuid)
        enable_language(RC)
        add_executable(icon EXCLUDE_FROM_ALL src/interface/icon.cpp)
        target_compile_features(icon PRIVATE cxx_std_20)
        set_target_properties(icon PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/icon")
        set(iconFile "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/icon/icon.ico")
        set(iconResource "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/interface.rc")
        add_custom_command(OUTPUT "${iconFile}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/icon"
            COMMAND $<TARGET_FILE:icon> "${iconFile}"
            DEPENDS icon src/interface/icon.hpp src/interface/icons.inc
            VERBATIM)
        file(GENERATE OUTPUT "${iconResource}" CONTENT "1 ICON \"${iconFile}\"\n")
        set_source_files_properties("${iconResource}" PROPERTIES OBJECT_DEPENDS "${iconFile}")
        target_sources(interface PRIVATE "${iconResource}" "${iconFile}")
    endif()
    set_target_properties(interface process local online custom PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/interface"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/interface")
    add_custom_command(TARGET interface POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/ATTRIBUTION.txt"
            "$<TARGET_FILE_DIR:interface>/ATTRIBUTION.txt"
        VERBATIM)
endif()
