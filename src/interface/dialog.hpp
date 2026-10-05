// file selection: use the desktop's file chooser to supply paths to configuration fields.
#pragma once

#include <optional>
#include <string>

struct GLFWwindow;

namespace NEBULA {
    // cancellation returns an empty optional. a failure also supplies an error message.
    std::optional<std::string> chooseFile(GLFWwindow* window, bool program, std::string& error);
}
