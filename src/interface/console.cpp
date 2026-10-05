#include "console.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace NEBULA {
    Command parseCommand(std::string_view line) {
        std::istringstream stream{std::string(line)};
        std::string name, extra;
        if (!(stream >> name)) return Command::None;
        if (stream >> extra) return Command::Invalid;
        for (char& letter : name) {
            letter = static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
        }
        if (name == "help") return Command::Help;
        if (name == "pause") return Command::Pause;
        if (name == "resume") return Command::Resume;
        if (name == "step") return Command::Step;
        if (name == "restart") return Command::Restart;
        if (name == "quit") return Command::Quit;
        return Command::Invalid;
    }

    bool Console::insert(std::string_view text) {
        if (text.size() > 256 - input.size()) {
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
        return true;
    }

    void Console::edit(Edit key) {
        switch (key) {
            case Edit::Left: if (cursor > 0) --cursor; break;
            case Edit::Right: if (cursor < input.size()) ++cursor; break;
            case Edit::Home: cursor = 0; break;
            case Edit::End: cursor = input.size(); break;
            case Edit::Backspace:
                if (cursor > 0) input.erase(--cursor, 1);
                break;
            case Edit::Delete:
                if (cursor < input.size()) input.erase(cursor, 1);
                break;
            case Edit::Clear:
                input.clear();
                cursor = 0;
                break;
        }
    }

    Command Console::submit() {
        Command command = parseCommand(input);
        edit(Edit::Clear);
        return command;
    }
}
