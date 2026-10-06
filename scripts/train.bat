@echo off
setlocal
pushd "%~dp0.." || exit /b 1
python src\agents\nebula\train.py %*
set "result=%errorlevel%"
popd
exit /b %result%
