#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace NEBULA {
    inline constexpr int maxWindowLevel = 10; // the highest window size level selects full screen.
    enum class Command { None, Help, Back, Quit, Turn, Win, Auto, Invalid };
    enum class Edit { Left, Right, Home, End, Backspace, Delete, Clear };

    struct ParsedCommand {
        Command type = Command::None;
        int value = 0;
    };

    // translate one line into a command. the caller decides how to execute it.
    ParsedCommand parseCommand(std::string_view line);

    struct TextInput {
        std::string input;
        std::string feedback;
        std::size_t cursor = 0;

        // keep each insertion on one line and within the input length limit.
        bool insert(std::string_view text);
        void edit(Edit key);
    };

    struct Console : TextInput {
        bool opened = false;
        ParsedCommand submit();
    };
}
