#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace NEBULA {
    enum class Command { None, Help, Pause, Resume, Step, Restart, Quit, Invalid };
    enum class Edit { Left, Right, Home, End, Backspace, Delete, Clear };

    // translate one line into a command. the caller decides how to execute it.
    Command parseCommand(std::string_view line);

    struct Console {
        bool opened = false;
        std::string input;
        std::string feedback;
        std::size_t cursor = 0;

        // keep each insertion on one line and within the input length limit.
        bool insert(std::string_view text);
        void edit(Edit key);
        Command submit();
    };
}
