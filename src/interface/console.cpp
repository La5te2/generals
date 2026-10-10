#include "console.hpp"
#include "manual.hpp"
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
        if (name == "rd") {
            stream >> std::ws;
            if (stream.get() != '"') return {Command::Invalid};
            std::string path;
            if (!std::getline(stream, path, '"') || stream.eof() || stream >> extra) return {Command::Invalid};
            return {Command::Rd, 0, 0, path};
        }
        if (name == "win") {
            std::string wide, high;
            int width = 0, height = 0;
            if (!(stream >> wide >> high) || stream >> extra) return {Command::Invalid};
            auto integer = [](const std::string& text, int& value) {
                auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
                return error == std::errc{} && end == text.data() + text.size();
            };
            if (!integer(wide, width) || !integer(high, height)) return {Command::Invalid};
            if (width == 0 && height == 0) return {Command::Win, 0, 0};
            if (width < minWindowWidth || width > maxWindowWidth || height < minWindowHeight || height > maxWindowHeight)
                return {Command::Invalid};
            return {Command::Win, width, height};
        }
        if (name == "turn" || name == "auto") {
            std::string number;
            int value = 0;
            if (!(stream >> number) || stream >> extra) return {Command::Invalid};
            auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), value);
            if (error != std::errc{} || end != number.data() + number.size()) return {Command::Invalid};
            if (name == "auto") {
                if (value != 0 && value != 1) return {Command::Invalid};
                return {Command::Auto, value};
            }
            if (value <= 0) return {Command::Invalid};
            return {Command::Turn, value};
        }
        if (stream >> extra) return {Command::Invalid};
        if (name == "help") return {Command::Help};
        if (name == "man") return {Command::Man};
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

    // preserve paragraphs and blank lines, then wrap each paragraph to the available text width.
    std::vector<std::string_view> Console::lines(std::size_t columns) const {
        std::vector<std::string_view> result;
        std::string_view remaining = manual ? Manual::text : std::string_view(feedback);
        columns = std::max(std::size_t{1}, columns);
        while (!remaining.empty()) {
            auto end = remaining.find('\n');
            auto paragraph = remaining.substr(0, end);
            if (paragraph.empty()) result.push_back({});
            while (!paragraph.empty()) {
                auto length = std::min(columns, paragraph.size());
                if (length < paragraph.size()) {
                    auto space = paragraph.rfind(' ', length);
                    if (space != std::string_view::npos && space > 0) length = space;
                }
                result.push_back(paragraph.substr(0, length));
                paragraph.remove_prefix(length);
                while (paragraph.starts_with(' ')) paragraph.remove_prefix(1);
            }
            if (end == std::string_view::npos) break;
            remaining.remove_prefix(end + 1);
        }
        return result;
    }

    bool Console::scroll(int amount, std::size_t columns, std::size_t rows) {
        auto count = lines(columns).size();
        auto limit = count > rows ? count - rows : 0;
        auto current = static_cast<std::ptrdiff_t>(std::min(first, limit));
        auto next = static_cast<std::size_t>(std::clamp(current + amount, std::ptrdiff_t{0}, static_cast<std::ptrdiff_t>(limit)));
        bool changed = first != next;
        first = next;
        return changed;
    }

    ParsedCommand Console::submit() {
        ParsedCommand command = parseCommand(input);
        edit(Edit::Clear);
        manual = false;
        first = 0;
        return command;
    }
}
