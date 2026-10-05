@echo off
setlocal
pushd "%~dp0.." || exit /b 1

rem keep the package location across Visual Studio environment setup.
set "vcpkgRoot=%VCPKG_ROOT%"
if not defined vcpkgRoot for /f "delims=" %%P in ('where vcpkg 2^>nul') do if not defined vcpkgRoot set "vcpkgRoot=%%~dpP"

rem discover the x64 MSVC toolchain used by the Windows dependencies.
set "vswhere="
for /f "delims=" %%P in ('where vswhere 2^>nul') do if not defined vswhere set "vswhere=%%P"
if not defined vswhere if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" set "vswhere=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "visualStudio="
if defined vswhere for /f "usebackq delims=" %%P in (`"%vswhere%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "visualStudio=%%P"
if not defined visualStudio (
    echo MSVC x64 build tools were not found.
    popd
    exit /b 1
)
call "%visualStudio%\Common7\Tools\VsDevCmd.bat" -no_logo -arch=x64 -host_arch=x64
if errorlevel 1 goto finish
if defined vcpkgRoot set "VCPKG_ROOT=%vcpkgRoot%"

set "ninja="
for /f "delims=" %%P in ('where ninja 2^>nul') do if not defined ninja set "ninja=%%P"
if not defined ninja if exist "%visualStudio%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" set "ninja=%visualStudio%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if not defined ninja (
    echo Ninja was not found. Add Ninja to PATH.
    popd
    exit /b 1
)

set "toolchain="
if defined VCPKG_ROOT set toolchain="-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"

rem use the same encoding for compiler output and CMake's header dependency detection.
set "codepage="
for /f "tokens=2 delims=:" %%P in ('chcp') do set "codepage=%%P"
chcp 65001 >nul

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_MANIFEST_INSTALL=OFF "-DCMAKE_MAKE_PROGRAM=%ninja%" %toolchain% %*
if errorlevel 1 goto finish
cmake --build build --parallel

:finish
set "result=%errorlevel%"
if defined codepage chcp %codepage% >nul
popd
exit /b %result%
