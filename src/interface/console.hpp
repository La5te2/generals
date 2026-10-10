#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace NEBULA {
    inline constexpr int minWindowWidth = 720, minWindowHeight = 560;
    inline constexpr int maxWindowWidth = 7680, maxWindowHeight = 4320;
    enum class Command { None, Help, Man, Back, Quit, Turn, Win, Auto, Rd, Invalid };
    enum class Edit { Left, Right, Home, End, Backspace, Delete, Clear };

    struct ParsedCommand {
        Command type = Command::None;
        int value = 0;
        int height = 0; // Win uses value as width; other commands leave height unused.
        std::string text;
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
        bool manual = false;
        std::size_t first = 0;
        std::vector<std::string_view> lines(std::size_t columns) const;
        bool scroll(int amount, std::size_t columns, std::size_t rows);
        ParsedCommand submit();
    };
}
