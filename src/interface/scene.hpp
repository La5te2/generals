// shared page definitions: keep input handling and drawing consistent about page state and control positions.
#pragma once

#include "console.hpp"
#include "font.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>

struct Rect {
    float x, y, width, height;

    bool contains(double px, double py) const {
        return px >= x && px < x + width && py >= y && py < y + height;
    }
};

enum class Tool { None, Backward, Playback, Forward, Stop, Reset };

// page status and layout definition
namespace NEBULA {
    // configuration state and layout are shared by interface.cpp's input handling and renderer.cpp's drawing.
    enum class Scene { Home, Local, Online, Replay };
    inline constexpr std::array tools{Tool::Backward, Tool::Playback, Tool::Forward, Tool::Stop, Tool::Reset};

    // drawing, mouse input and shortcuts share the same availability rules.
    struct BoardControls {
        Scene scene = Scene::Local;
        // human is true while an unfinished match has a keyboard-and-mouse player.
        bool active = false, running = false, human = false;
        Tool hover = Tool::None;
        bool previous = false; // the replay reader supplies whether an earlier position is available.

        bool spectator() const { return (scene == Scene::Local || scene == Scene::Replay) && !human; }

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
        // online fields stay in memory. the user ID is masked while drawing.
        std::array<TextInput, 8> fields; // username, user ID, replay path, red command, blue command, online command, room, proxy.
        std::array<std::array<TextInput, 4>, 3> accounts;
        void selectServer(int selected) {
            accounts[server] = {fields[0], fields[1], fields[6], fields[7]};
            server = selected;
            fields[0] = accounts[server][0];
            fields[1] = accounts[server][1];
            fields[6] = accounts[server][2];
            fields[7] = accounts[server][3];
        }
        int milliseconds = 500; // local and replay half-turn interval, retained between sessions.
        int focus = -1;
        int fileHover = -1;
        std::string message;
        Timer::time_point messageTime{};

        // a blank or whitespace-only player command selects keyboard and mouse control.
        // fields 3 and 4 select the local red and blue players. field 5 selects the online player.
        bool humanPlayer(int field) const {
            return (field == 3 || field == 4 || field == 5) &&
                fields[field].input.find_first_not_of(" \t\r\n") == std::string::npos;
        }

        // every notification starts a fresh display interval, including repeated text.
        void notify(std::string_view text, Timer::time_point now = Timer::now()) {
            message = text;
            messageTime = now;
        }

        // store the current message and its display time, for fading out after a few seconds
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

    inline bool hasFileButton(Scene scene, int index) {
        return (scene == Scene::Local && (index == 3 || index == 4)) ||
               (scene == Scene::Online && index == 5) || (scene == Scene::Replay && index == 2);
    }

    inline Rect fieldRow(Scene scene, int index, int width, int height) {
        int group = scene == Scene::Local ? index - 3 : scene == Scene::Replay ? 0 : index >= 6 ? index - 2 : index == 5 ? 1 : index + 2;
        return formControl(scene, group, width, height);
    }

    // a file button occupies one square at the row's right edge, separated from the input by 8 scaled pixels.
    inline Rect inputField(Scene scene, int index, int width, int height) {
        Rect row = fieldRow(scene, index, width, height);
        if (hasFileButton(scene, index)) row.width -= row.height + 8 * contentScale(width, height);
        return row;
    }

    inline Rect fileButton(Scene scene, int index, int width, int height) {
        Rect row = fieldRow(scene, index, width, height);
        return {row.x + row.width - row.height, row.y, row.height, row.height};
    }

    // the same order drives mouse focus and Tab navigation.
    inline std::array<int, 5> inputOrder(Scene scene) {
        if (scene == Scene::Local) return {3, 4, -1, -1, -1};
        if (scene == Scene::Online) return {5, 0, 1, 6, 7};
        if (scene == Scene::Replay) return {2, -1, -1, -1, -1};
        return {-1, -1, -1, -1, -1};
    }

    inline Rect startButton(int width, float scale) {
        return {width - 16.0f - 180 * scale, 12 - 4 * scale, 180 * scale, 36 * scale};
    }
    inline Rect backButton(float scale) { return {16, 12, 28 * scale, 28 * scale}; }
    // operation feedback sits below the top bar, independently of the centered form.
    inline Rect messageArea(int width, float scale) { return {16, 40 + 28 * scale, width - 32.0f, 36 * scale}; }
}
