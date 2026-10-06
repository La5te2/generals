#include "console.hpp"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <sstream>

namespace NEBULA {
    ParsedCommand parseCommand(std::string_view line) {
        std::istringstream stream{std::string(line)};
        std::string name, extra;
        if (!(stream >> name)) return {};
        for (char& letter : name) {
            letter = static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
        }
        if (name == "turn" || name == "win" || name == "auto") {
            std::string number;
            int value = 0;
            if (!(stream >> number) || stream >> extra) return {Command::Invalid};
            auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), value);
            if (error != std::errc{} || end != number.data() + number.size()) return {Command::Invalid};
            if (name == "auto") {
                if (value != 0 && value != 1) return {Command::Invalid};
                return {Command::Auto, value};
            }
            if (name == "win") {
                if (value < 0 || value > maxWindowLevel) return {Command::Invalid};
                return {Command::Win, value};
            }
            if (value <= 0) return {Command::Invalid};
            return {Command::Turn, value};
        }
        if (stream >> extra) return {Command::Invalid};
        if (name == "help") return {Command::Help};
        if (name == "back") return {Command::Back};
        if (name == "quit") return {Command::Quit};
        return {Command::Invalid};
    }

    bool TextInput::insert(std::string_view text) {
        if (input.size() > 256 || text.size() > 256 - input.size()) {
            feedback = "Input limit: 256 characters";
            return false;
        }
        // validate the whole insertion so pasted line breaks keep the current input unchanged.
        if (std::any_of(text.begin(), text.end(), [](unsigned char letter) {
            return letter < 32 || letter > 126;
        })) {
            feedback = "Use one line of ASCII text";
            return false;
        }
        input.insert(cursor, text);
        cursor += text.size();
        feedback.clear();
        return true;
    }

    void TextInput::edit(Edit key) {
        feedback.clear();
        // native file dialogs can supply UTF-8 paths. cursor movement and deletion preserve whole characters.
        auto previous = [&](std::size_t position) {
            if (position > 0) --position;
            while (position > 0 && (static_cast<unsigned char>(input[position]) & 0xc0) == 0x80) --position;
            return position;
        };
        auto next = [&](std::size_t position) {
            if (position < input.size()) ++position;
            while (position < input.size() && (static_cast<unsigned char>(input[position]) & 0xc0) == 0x80) ++position;
            return position;
        };
        switch (key) {
            case Edit::Left: cursor = previous(cursor); break;
            case Edit::Right: cursor = next(cursor); break;
            case Edit::Home: cursor = 0; break;
            case Edit::End: cursor = input.size(); break;
            case Edit::Backspace:
                if (cursor > 0) {
                    auto position = previous(cursor);
                    input.erase(position, cursor - position);
                    cursor = position;
                }
                break;
            case Edit::Delete:
                if (cursor < input.size()) input.erase(cursor, next(cursor) - cursor);
                break;
            case Edit::Clear:
                input.clear();
                cursor = 0;
                break;
        }
    }

    ParsedCommand Console::submit() {
        ParsedCommand command = parseCommand(input);
        edit(Edit::Clear);
        return command;
    }
}
