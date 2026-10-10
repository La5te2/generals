// shared page definitions: keep input handling and drawing consistent about page state and control positions.
// Own form values, focus, notification state, layout rectangles and control availability; do not run sessions or draw.
#pragma once

#include "console.hpp"
#include "font.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <stdexcept>

struct Rect {
    float x, y, width, height;

    bool contains(double px, double py) const {
        return px >= x && px < x + width && py >= y && py < y + height;
    }
};

enum class Tool { None, Backward, Playback, Forward, Stop, Reset };

inline Rect viewButton(int view, float scale) {
    return {16 + (44 + view * 64) * scale, 12, 64 * scale, 28 * scale};
}
inline Rect toolButton(Tool tool, int width, float scale) {
    float offset = 28 + (static_cast<int>(Tool::Reset) - static_cast<int>(tool)) * 36.0f;
    return {width - 16.0f - offset * scale, 12, 28 * scale, 28 * scale};
}

// page status and layout definition
namespace NEBULA {
    // configuration state and layout are shared by interface.cpp's input handling and renderer.cpp's drawing.
    enum class Scene { Home, Local, Online, Replay };
    enum class Field { None, Username, Identity, Replay, Red, Blue, Player, Room, Address, Proxy };
    inline constexpr std::array tools{Tool::Backward, Tool::Playback, Tool::Forward, Tool::Stop, Tool::Reset};

    // drawing, mouse input and shortcuts share the same availability rules.
    struct BoardControls {
        Scene scene = Scene::Local;
        // human is true while an unfinished match has a keyboard-and-mouse player.
        bool active = false, running = false, human = false;
        Tool hover = Tool::None;
        bool previous = false; // the replay reader supplies whether an earlier position is available.
        bool review = false; // an ended network match supplies all three final perspectives.

        bool spectator() const { return ((scene == Scene::Local || scene == Scene::Replay) && !human) || review; }

        bool enabled(Tool tool) const {
            if (tool == Tool::Stop) return active && scene != Scene::Replay;
            if (tool == Tool::Reset) return !active;
            if (tool == Tool::Backward && scene == Scene::Replay) return !running && previous;
            if (!active || scene == Scene::Online || scene == Scene::Home) return false;
            if (tool == Tool::Playback) return true;
            if (running || human) return false;
            if (tool == Tool::Backward) return scene == Scene::Replay && previous;
            if (tool == Tool::Forward) return scene == Scene::Local || scene == Scene::Replay;
            return false;
        }
    };
    // save the current page, form content, and focus
    struct Setup {
        using Timer = std::chrono::steady_clock;
        static constexpr double messageHold = 2, messageFade = 1; // seconds.

        Scene scene = Scene::Home;
        int server = 0; // BOT, MAIN, LAN.
        bool lan() const { return server == 2; }
        std::filesystem::path directory; // session-wide RD destination; empty disables local recording.
        struct Account { TextInput username, identity, room, proxy; };
        std::array<Account, 3> accounts;
        TextInput replay, red, blue, player, address;
        void selectServer(int selected) { if (selected >= 0 && selected < 3) server = selected; }
        template<class Self> static auto& access(Self& self, Field field) {
            switch (field) {
                case Field::Username: return self.accounts[self.server].username;
                case Field::Identity: return self.accounts[self.server].identity;
                case Field::Room: return self.accounts[self.server].room;
                case Field::Address: return self.address;
                case Field::Proxy: return self.accounts[self.server].proxy;
                case Field::Replay: return self.replay;
                case Field::Red: return self.red;
                case Field::Blue: return self.blue;
                case Field::Player: return self.player;
                default: throw std::out_of_range("No configuration field selected");
            }
        }
        TextInput& field(Field id) { return access(*this, id); }
        const TextInput& field(Field id) const { return access(*this, id); }
        int milliseconds = 500; // local and replay half-turn interval, retained between sessions.
        Field focus = Field::None;
        Field fileHover = Field::None;
        std::string message;
        Timer::time_point messageTime{};

        // a blank or whitespace-only player command selects keyboard and mouse control.
        bool humanPlayer(Field id) const {
            return (id == Field::Red || id == Field::Blue || id == Field::Player) &&
                field(id).input.find_first_not_of(" \t\r\n") == std::string::npos;
        }

        // every notification starts a fresh display interval, including repeated text.
        void notify(std::string_view text, Timer::time_point now = Timer::now()) {
            message = text;
            messageTime = now;
        }

        // hold at full opacity, then fade linearly using elapsed real time.
        float messageOpacity(Timer::time_point now = Timer::now()) const {
            if (message.empty()) return 0;
            double elapsed = std::chrono::duration<double>(now - messageTime).count();
            return static_cast<float>(std::clamp((messageHold + messageFade - elapsed) / messageFade, 0.0, 1.0));
        }

        // sleep through the hold period, then request regular updates during the fade.
        double messageWait(Timer::time_point now = Timer::now()) const {
            double elapsed = std::chrono::duration<double>(now - messageTime).count();
            return std::max(messageHold - elapsed, 1.0 / 60);
        }
    };

    // drawing and mouse input share these dimensions in logical window coordinates.
    inline float contentScale(int width, int height) {
        // 960 x 800 is the reference size. keep the central controls between 90% and 175%.
        return std::clamp(std::min(width / 960.0f, height / 800.0f), .9f, 1.75f);
    }

    // top and bottom controls scale with window height, between 100% and 150% of their reference size.
    inline float barScale(int height) { return std::clamp(height / 800.0f, 1.0f, 1.5f); }

    struct ConsoleLayout {
        Rect input;
        std::size_t columns, rows;
    };

    // drawing and scrolling use the same wrapping width and visible row count.
    // the output fits between the top controls and the fixed input line above the turn counter.
    inline ConsoleLayout consoleLayout(int width, int height) {
        float scale = barScale(height);
        float top = height - 44 - 36 * scale;
        auto columns = static_cast<std::size_t>(std::max(1.0f, (width - 32 - 32 * scale) / (PixelFont::advance * 1.5f * scale)));
        auto rows = static_cast<std::size_t>(std::max(1.0f, (top - 64 - 8 * scale) / (22 * scale)));
        return {{16, top, width - 32.0f, 36 * scale}, columns, rows};
    }

    // align the score table with the right edge of the playback controls.
    inline Rect scoreArea(int width, int height) {
        float scale = barScale(height);
        float span = std::min(172 * scale, std::max(0.0f, width - 32.0f));
        return {width - 16.0f - span, 12 + 36 * scale, span, 84 * scale};
    }

    inline Rect scoreNamesArea(int width, int height) {
        Rect scores = scoreArea(width, height);
        float scale = barScale(height);
        return {scores.x, scores.y + scores.height + 10 * scale, scores.width, 64 * scale};
    }

    // center the board in the window. use space below the score table when the right margin is too narrow.
    // drawing and board input use this same rectangle.
    inline Rect boardArea(int rows, int cols, int width, int height) {
        float top = 32 + 28 * barScale(height);
        if (rows <= 0 || cols <= 0 || height <= top + 40 || width <= 32) return {};
        auto centered = [&](float upper) {
            float available = height - upper - 40;
            if (available <= 0) return Rect{};
            float cell = std::min((width - 32.0f) / cols, available / rows);
            return Rect{(width - cell * cols) / 2, upper + (available - cell * rows) / 2, cell * cols, cell * rows};
        };
        Rect board = centered(top);
        Rect scores = scoreArea(width, height), names = scoreNamesArea(width, height);
        float lower = names.y + names.height + 12;
        if (board.x + board.width + 12 > scores.x && board.y < lower) board = centered(lower);
        return board;
    }

    inline Rect menuTitle(int width, int height) {
        float scale = contentScale(width, height);
        float span = 400 * scale;
        // the middle button's center sits 158 scaled pixels below the title's top.
        return {(width - span) / 2, height / 2.0f - 158 * scale, span, 36 * scale};
    }

    inline Rect menuButton(int index, int width, int height) {
        Rect title = menuTitle(width, height);
        float scale = contentScale(width, height);
        return {title.x, title.y + (64 + index * 68) * scale, title.width, 52 * scale};
    }

    inline Rect formControl(Scene scene, int index, int width, int height) {
        int groups = scene == Scene::Online ? 6 : scene == Scene::Local ? 2 : 1;
        // reserve 24 px for each label and 36 px for its control. six online rows use a tighter, uniform gap.
        float scale = contentScale(width, height);
        float spacing = scene == Scene::Online ? 80.0f : 88.0f;
        float span = ((groups - 1) * spacing + 60) * scale;
        return {(width - 600 * scale) / 2, (height - span) / 2 + (index * spacing + 24) * scale,
                600 * scale, 36 * scale};
    }

    inline Rect choiceButton(Rect row, int choice, int count) {
        float gap = 4 * row.height / 36;
        float span = (row.width - (count - 1) * gap) / count;
        return {row.x + (span + gap) * choice, row.y, span, row.height};
    }

    // This order determines both row positions and keyboard navigation.
    inline std::array<Field, 5> inputOrder(const Setup& setup) {
        if (setup.scene == Scene::Local) return {Field::Red, Field::Blue};
        if (setup.scene == Scene::Online) {
            if (setup.lan()) return {Field::Player, Field::Username, Field::Room, Field::Address, Field::Proxy};
            return {Field::Player, Field::Username, Field::Identity, Field::Room, Field::Proxy};
        }
        if (setup.scene == Scene::Replay) return {Field::Replay};
        return {};
    }
    inline bool hasFileButton(Field field) {
        return field == Field::Red || field == Field::Blue || field == Field::Player || field == Field::Replay;
    }
    inline std::string_view fieldLabel(const Setup& setup, Field field) {
        switch (field) {
            case Field::Username: return "USERNAME";
            case Field::Identity: return "USER ID";
            case Field::Room: return setup.lan() ? "ROOM ID" : "PRIVATE ROOM";
            case Field::Address: return "IP";
            case Field::Proxy: return "PROXY";
            case Field::Red: return "RED PLAYER";
            case Field::Blue: return "BLUE PLAYER";
            case Field::Player: return "PLAYER";
            case Field::Replay: return "REPLAY FILE";
            default: return "";
        }
    }
    inline Rect fieldRow(const Setup& setup, Field field, int width, int height) {
        auto order = inputOrder(setup);
        int group = static_cast<int>(std::find(order.begin(), order.end(), field) - order.begin()) + (setup.scene == Scene::Online ? 1 : 0);
        return formControl(setup.scene, group, width, height);
    }

    // a file button occupies one square at the row's right edge, separated from the input by 8 scaled pixels.
    inline Rect inputField(const Setup& setup, Field field, int width, int height) {
        Rect row = fieldRow(setup, field, width, height);
        if (hasFileButton(field)) row.width -= row.height + 8 * contentScale(width, height);
        return row;
    }

    inline Rect fileButton(const Setup& setup, Field field, int width, int height) {
        Rect row = fieldRow(setup, field, width, height);
        return {row.x + row.width - row.height, row.y, row.height, row.height};
    }

    inline Rect startButton(int width, float scale) {
        return {width - 16.0f - 180 * scale, 12 - 4 * scale, 180 * scale, 36 * scale};
    }
    inline Rect backButton(float scale) { return {16, 12, 28 * scale, 28 * scale}; }
    // Configuration feedback sits below the top bar; board feedback sits below the player names and tooltips.
    inline Rect messageArea(int width, int height, bool board) {
        float scale = barScale(height);
        if (!board) return {16, 40 + 28 * scale, width - 32.0f, 36 * scale};
        Rect names = scoreNamesArea(width, height);
        return {names.x, names.y + names.height + 44 * scale, names.width, 36 * scale};
    }
}
