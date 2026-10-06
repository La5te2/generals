// file selection: use the desktop's file chooser to supply paths to configuration fields.
#pragma once

#include <optional>
#include <string>

struct GLFWwindow;

namespace NEBULA {
    enum class PathKind { Program, Replay, Directory };
    // cancellation returns an empty optional. a failure also supplies an error message.
    std::optional<std::string> choosePath(GLFWwindow* window, PathKind kind, std::string& error);
}
