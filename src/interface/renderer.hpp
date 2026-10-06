#pragma once

#include "engine/observe.hpp"
#include "controller.hpp"
#include "console.hpp"
#include "scene.hpp"
#include <glad/glad.h>
#include <span>
#include <string_view>
#include <vector>

// drawing and mouse input use the same logical window coordinates.
inline Rect viewButton(int view, float scale) {
    return {16 + (44 + view * 64) * scale, 12, 64 * scale, 28 * scale};
}
inline Rect toolButton(Tool tool, int width, float scale) {
    float offset = 28 + (static_cast<int>(Tool::Reset) - static_cast<int>(tool)) * 36.0f;
    return {width - 16.0f - offset * scale, 12, 28 * scale, 28 * scale};
}

class Renderer {
public:
    Renderer() = default;
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    ~Renderer();

    // create GPU resources after GLAD loads. destroy them before the window closes.
    bool init();
    void draw(const Observation& view, int perspective, int width, int height,
              const NEBULA::BoardControls& controls, const NEBULA::Setup& setup,
              const NEBULA::Console& console, const NEBULA::Controller& controller, std::span<const Action> queued);
    void drawSetup(const NEBULA::Setup& setup, int width, int height, const NEBULA::Console& console);

private:
    struct Color { float r, g, b, a = 1; };
    struct Point { float x, y; };
    struct Vertex { Point position; Color color; };

    void triangle(Point a, Point b, Point c, Color color);
    void rectangle(Rect rect, Color color);
    void line(Point a, Point b, float thickness, Color color);
    void text(std::string_view value, float x, float y, float scale, Color color);
    void icon(ViewTerrain terrain, Rect bounds, Color color);
    void drawSelection(const NEBULA::Controller& controller, std::span<const Action> queued, Rect board, int cols);
    void drawConsole(const NEBULA::Console& console, int width, int height, float textScale);
    void drawMessage(const NEBULA::Setup& setup, int width, int height, bool board = false);
    void drawScores(const Observation& view, const NEBULA::Setup& setup, int width, int height);
    void drawTools(const NEBULA::BoardControls& controls, int width, float scale);
    void label(std::string_view value, Rect bounds, Color color, float scale = 1.5f);
    void button(std::string_view value, Rect bounds, bool selected = false, bool enabled = true, float scale = 1);
    void input(const NEBULA::TextInput& field, Rect bounds, bool focused, bool masked = false, float scale = 1);
    void back(float scale);
    void flush(int width, int height);

    GLuint program = 0, vao = 0, vbo = 0;
    GLint canvas = -1;
    std::vector<Vertex> vertices;
};
