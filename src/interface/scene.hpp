#pragma once

#include "console.hpp"
#include <algorithm>
#include <array>

struct Rect {
    float x, y, width, height;

    bool contains(double px, double py) const {
        return px >= x && px < x + width && py >= y && py < y + height;
    }
};

namespace NEBULA {
    enum class Scene { Home, Local, Online, Replay };

    struct Setup {
        Scene scene = Scene::Home;
        bool mainServer = false;
        // online fields stay in memory. the user ID is masked while drawing.
        std::array<TextInput, 3> fields; // username, user ID, replay path.
        int focus = -1;
        std::string message;
    };

    // drawing and mouse input share these dimensions in logical window coordinates.
    inline Rect menuTitle(int width, int height) {
        float span = std::min(400.0f, width - 64.0f);
        return {(width - span) / 2, (height - 252.0f) / 2, span, 36};
    }

    inline Rect menuButton(int index, int width, int height) {
        Rect title = menuTitle(width, height);
        return {title.x, title.y + 64 + index * 68, title.width, 52};
    }

    inline Rect formRow(float top, int width) {
        float span = std::min(600.0f, width - 64.0f);
        return {(width - span) / 2, top, span, 36};
    }

    inline Rect formControl(Scene scene, int index, int width, int height) {
        int groups = scene == Scene::Online ? 4 : scene == Scene::Local ? 2 : 1;
        // each group has a 24 px label area, a 36 px control and a 28 px gap to the next group.
        float span = groups * 88.0f - 28;
        return formRow((height - span) / 2 + index * 88 + 24, width);
    }

    inline Rect choiceButton(Rect row, int choice, int count) {
        float span = (row.width - (count - 1) * 4) / count;
        return {row.x + (span + 4) * choice, row.y, span, row.height};
    }

    inline Rect inputField(Scene scene, int index, int width, int height) {
        return formControl(scene, scene == Scene::Replay ? 0 : index + 2, width, height);
    }

    inline Rect startButton(int width) { return {width - 196.0f, 8, 180, 36}; }
    inline Rect backButton() { return {16, 12, 28, 28}; }
    inline Rect setupButton(int width) { return {width - 172.0f, 12, 156, 28}; }
}
