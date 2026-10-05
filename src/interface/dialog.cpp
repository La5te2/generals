// desktop file dialogs: select an existing file while keeping platform-specific APIs outside page and input logic.
#include "dialog.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <array>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <commdlg.h>
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
    std::optional<std::string> chooseFile(GLFWwindow* window, bool program, std::string& error) {
        error.clear();
#ifdef _WIN32
        std::array<wchar_t, 32768> path{};
        OPENFILENAMEW dialog{};
        dialog.lStructSize = sizeof(dialog);
        dialog.hwndOwner = glfwGetWin32Window(window);
        dialog.lpstrFile = path.data();
        dialog.nMaxFile = static_cast<DWORD>(path.size());
        dialog.lpstrTitle = program ? L"Select player program" : L"Select replay file";
        dialog.lpstrFilter = program ? L"Programs (*.exe)\0*.exe\0All files\0*.*\0" : L"All files\0*.*\0";
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
        const char* script = program
            ? "try\nreturn POSIX path of (choose file with prompt \"Select player program\")\non error number -128\nreturn \"\"\nend try"
            : "try\nreturn POSIX path of (choose file with prompt \"Select replay file\")\non error number -128\nreturn \"\"\nend try";
        const std::vector<std::vector<const char*>> choices{{"osascript", "-e", script}};
#else
        const std::vector<std::vector<const char*>> choices{
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
