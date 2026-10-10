// file selection interface: let configuration pages request a player-program or replay path without depending on desktop APIs.
// choosePath returns the selected path, no value on cancellation, or no value with an error message on failure.
// the caller fills the input field; starting programs and loading replays remain separate operations.
#pragma once

#include <optional>
#include <string>

struct GLFWwindow;

namespace NEBULA {
    enum class PathKind { Program, Replay };
    // cancellation returns an empty optional. a failure also supplies an error message.
    std::optional<std::string> choosePath(GLFWwindow* window, PathKind kind, std::string& error);
}
