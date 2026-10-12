// file selection interface: let configuration pages request a player-program or replay path without depending on desktop APIs.
// the caller fills the input field; starting programs and loading replays remain separate operations.
#pragma once

#include <optional>
#include <string>

struct GLFWwindow;

namespace NEBULA {
    enum class PathKind { Program, Replay };
    // Open a modal picker and return a UTF-8 path. Cancellation returns no value and clears error; failure sets error.
    // This call blocks while the dialog is open; it does not validate program execution or replay contents.
    std::optional<std::string> choosePath(GLFWwindow* window, PathKind kind, std::string& error);
}
