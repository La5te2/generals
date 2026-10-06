// desktop file dialogs: select an existing file while keeping platform-specific APIs outside page and input logic.
#include "dialog.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <array>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <commdlg.h>
#include <shobjidl.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
extern char** environ;
#endif

namespace NEBULA {
    std::optional<std::string> choosePath(GLFWwindow* window, PathKind kind, std::string& error) {
        error.clear();
        bool program = kind == PathKind::Program;
        bool directory = kind == PathKind::Directory;
#ifdef _WIN32
        if (directory) {
            HRESULT initialized = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
            if (FAILED(initialized)) { error = "Directory dialog initialization failed"; return std::nullopt; }
            IFileOpenDialog* dialog = nullptr;
            HRESULT result = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog));
            std::optional<std::string> selected;
            if (SUCCEEDED(result)) {
                FILEOPENDIALOGOPTIONS flags{};
                result = dialog->GetOptions(&flags);
                if (SUCCEEDED(result)) result = dialog->SetOptions(flags | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR);
                if (SUCCEEDED(result)) result = dialog->SetTitle(L"Select recording directory");
                if (SUCCEEDED(result)) result = dialog->Show(glfwGetWin32Window(window));
                if (SUCCEEDED(result)) {
                    IShellItem* item = nullptr;
                    result = dialog->GetResult(&item);
                    if (SUCCEEDED(result)) {
                        wchar_t* path = nullptr;
                        result = item->GetDisplayName(SIGDN_FILESYSPATH, &path);
                        if (SUCCEEDED(result)) {
                            int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, path, -1, nullptr, 0, nullptr, nullptr);
                            if (size > 0) {
                                selected.emplace(size, '\0');
                                WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, path, -1, selected->data(), size, nullptr, nullptr);
                                selected->pop_back();
                            } else result = E_FAIL;
                            CoTaskMemFree(path);
                        }
                        item->Release();
                    }
                }
                dialog->Release();
            }
            CoUninitialize();
            if (FAILED(result) && result != HRESULT_FROM_WIN32(ERROR_CANCELLED)) error = "Directory selection failed";
            return selected;
        }
        std::array<wchar_t, 32768> path{};
        OPENFILENAMEW dialog{};
        dialog.lStructSize = sizeof(dialog);
        dialog.hwndOwner = glfwGetWin32Window(window);
        dialog.lpstrFile = path.data();
        dialog.nMaxFile = static_cast<DWORD>(path.size());
        dialog.lpstrTitle = program ? L"Select player program" : L"Select replay file";
        dialog.lpstrFilter = program ? L"Programs (*.exe)\0*.exe\0All files\0*.*\0" : L"Replay files (*.gior)\0*.gior\0All files\0*.*\0";
        dialog.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
        if (!GetOpenFileNameW(&dialog)) {
            DWORD code = CommDlgExtendedError();
            if (code) error = "File dialog failed (Windows error " + std::to_string(code) + ")";
            return std::nullopt;
        }
        int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, path.data(), -1, nullptr, 0, nullptr, nullptr);
        if (!length) { error = "Cannot read the selected path"; return std::nullopt; }
        std::string selected(length, '\0');
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, path.data(), -1, selected.data(), length, nullptr, nullptr);
        selected.pop_back();
        return selected;
#else
        (void)window;
        // invoke desktop helpers directly. file paths are returned through stdout rather than inserted into shell commands.
#ifdef __APPLE__
        const char* script = directory
            ? "try\nreturn POSIX path of (choose folder with prompt \"Select recording directory\")\non error number -128\nreturn \"\"\nend try"
            : program
            ? "try\nreturn POSIX path of (choose file with prompt \"Select player program\")\non error number -128\nreturn \"\"\nend try"
            : "try\nreturn POSIX path of (choose file with prompt \"Select replay file\")\non error number -128\nreturn \"\"\nend try";
        const std::vector<std::vector<const char*>> choices{{"osascript", "-e", script}};
#else
        const std::vector<std::vector<const char*>> choices = directory ? std::vector<std::vector<const char*>>{
            {"zenity", "--file-selection", "--directory", "--title=Select recording directory"},
            {"kdialog", "--getexistingdirectory", ".", "--title", "Select recording directory"}
        } : std::vector<std::vector<const char*>>{
            {"zenity", "--file-selection", program ? "--title=Select player program" : "--title=Select replay file"},
            {"kdialog", "--getopenfilename", ".", "--title", program ? "Select player program" : "Select replay file"}
        };
#endif
        for (const auto& command : choices) {
            std::array<int, 2> streams{};
            if (pipe(streams.data())) { error = "Cannot open file dialog output"; return std::nullopt; }
            if (fcntl(streams[0], F_SETFD, FD_CLOEXEC) == -1 || fcntl(streams[1], F_SETFD, FD_CLOEXEC) == -1) {
                close(streams[0]); close(streams[1]);
                error = "Cannot configure file dialog output"; return std::nullopt;
            }
            posix_spawn_file_actions_t files;
            int status = posix_spawn_file_actions_init(&files);
            if (status) {
                close(streams[0]); close(streams[1]);
                error = "Cannot initialize file dialog"; return std::nullopt;
            }
            status = posix_spawn_file_actions_adddup2(&files, streams[1], STDOUT_FILENO);
            if (!status) status = posix_spawn_file_actions_addopen(&files, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
            std::vector<char*> args;
            for (const char* argument : command) args.push_back(const_cast<char*>(argument));
            args.push_back(nullptr);
            pid_t child = -1;
            if (!status) status = posix_spawnp(&child, args[0], &files, nullptr, args.data(), environ);
            posix_spawn_file_actions_destroy(&files);
            close(streams[1]);
            if (status) {
                close(streams[0]);
                if (status == ENOENT) continue;
                error = "Cannot start file dialog"; return std::nullopt;
            }
            std::string selected;
            std::array<char, 1024> buffer{};
            bool failed = false;
            for (;;) {
                ssize_t count = read(streams[0], buffer.data(), buffer.size());
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) { failed = failed || count < 0; break; }
                if (selected.size() + static_cast<std::size_t>(count) <= 32768) selected.append(buffer.data(), count);
                else failed = true;
            }
            close(streams[0]);
            pid_t waited;
            do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
            if (waited < 0 || failed || !WIFEXITED(status)) {
                error = "Cannot read file dialog result"; return std::nullopt;
            }
            if (WEXITSTATUS(status) == 1) return std::nullopt;
            if (WEXITSTATUS(status) != 0) { error = "File dialog failed"; return std::nullopt; }
            if (!selected.empty() && selected.back() == '\n') selected.pop_back();
            if (selected.empty()) return std::nullopt;
            return selected;
        }
        error = "File chooser unavailable. Enter the path directly";
        return std::nullopt;
#endif
    }
}
