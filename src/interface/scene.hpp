// shared page definitions: keep input handling and drawing consistent about page state and control positions.
#pragma once

#include "console.hpp"
#include <algorithm>
#include <array>
#include <chrono>

struct Rect {
    float x, y, width, height;

    bool contains(double px, double py) const {
        return px >= x && px < x + width && py >= y && py < y + height;
    }
};

// page status and layout definition
namespace NEBULA {
    // configuration state and layout are shared by nebula.cpp's input handling and renderer.cpp's drawing.
    enum class Scene { Home, Local, Online, Replay };
    // save the current page, form content, and focus
    struct Setup {
        using Timer = std::chrono::steady_clock;
        static constexpr double messageHold = 2, messageFade = 1; // seconds.

        Scene scene = Scene::Home;
        bool mainServer = false;
        // online fields stay in memory. the user ID is masked while drawing.
        std::array<TextInput, 6> fields; // username, user ID, replay path, red command, blue command, online command.
        int milliseconds = 500; // local half-turn duration, retained between games.
        int focus = -1;
        int fileHover = -1;
        std::string message;
        Timer::time_point messageTime{};

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
        // 960 x 800 is the reference size. keep the central controls between 75% and 175%.
        return std::clamp(std::min(width / 960.0f, height / 800.0f), .75f, 1.75f);
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
        int groups = scene == Scene::Online ? 4 : scene == Scene::Local ? 2 : 1;
        // each group has a 24 px label area, a 36 px control and a 28 px gap to the next group.
        float scale = contentScale(width, height);
        float span = (groups * 88.0f - 28) * scale;
        return {(width - 600 * scale) / 2, (height - span) / 2 + (index * 88 + 24) * scale,
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
        int group = scene == Scene::Local ? index - 3 : scene == Scene::Replay ? 0 : index == 5 ? 1 : index + 2;
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
    inline std::array<int, 3> inputOrder(Scene scene) {
        if (scene == Scene::Local) return {3, 4, -1};
        if (scene == Scene::Online) return {5, 0, 1};
        if (scene == Scene::Replay) return {2, -1, -1};
        return {-1, -1, -1};
    }

    inline Rect startButton(int width) { return {width - 196.0f, 8, 180, 36}; }
    inline Rect backButton() { return {16, 12, 28, 28}; }
    inline Rect setupButton(int width) { return {width - 172.0f, 12, 156, 28}; }

    // operation feedback sits below the top bar, independently of the centered form.
    inline Rect messageArea(int width) { return {16, 68, width - 32.0f, 36}; }
}
