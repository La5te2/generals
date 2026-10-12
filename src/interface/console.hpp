// Console command syntax and editable text state shared by console and configuration fields.
// The application executes parsed commands; the renderer presents text using scene-defined layout.
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

    // Parse case-insensitive command names and validate arguments, without executing anything.
    // Empty input returns None; unknown commands, invalid arguments and extra tokens return Invalid.
    ParsedCommand parseCommand(std::string_view line);

    struct TextInput {
        std::string input;
        std::string feedback;
        std::size_t cursor = 0;

        // Insert printable ASCII at the cursor, up to 256 bytes total; reject the whole insertion on error.
        bool insert(std::string_view text);
        // Move/delete by complete UTF-8 characters so paths from native dialogs remain intact; clear stale feedback.
        void edit(Edit key);
    };

    struct Console : TextInput {
        bool opened = false;
        bool manual = false;
        std::size_t first = 0;
        // Wrap the manual or feedback without copying its text; returned views remain tied to that text's lifetime.
        std::vector<std::string_view> lines(std::size_t columns) const;
        // Clamp scrolling to the wrapped output and visible row count; return whether the first visible row changed.
        bool scroll(int amount, std::size_t columns, std::size_t rows);
        // Parse the current input, then clear it and reset output/manual scrolling; execution belongs to WindowState.
        ParsedCommand submit();
    };
}
