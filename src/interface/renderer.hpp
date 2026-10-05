#pragma once

#include "engine/observe.hpp"
#include "console.hpp"
#include "scene.hpp"
#include <glad/glad.h>
#include <string_view>
#include <vector>

// drawing and mouse input use the same logical window coordinates.
inline Rect viewButton(int view) { return {60.0f + view * 64.0f, 12, 64, 28}; }
enum class Tool { None, Playback, Step, Stop };

inline Rect toolButton(Tool tool, int width) {
    float offset = tool == Tool::Playback ? 116.0f : tool == Tool::Step ? 80.0f : 44.0f;
    return {static_cast<float>(width) - offset, 12, 28, 28};
}

class Renderer {
public:
    Renderer() = default;
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    ~Renderer();

    // create GPU resources after GLAD loads. destroy them before the window closes.
    bool init();
    void draw(const Observation& view, int perspective, int width, int height, bool running, bool active, Tool hover,
              const NEBULA::Console& console);
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
    void drawConsole(const NEBULA::Console& console, int width, int height);
    void label(std::string_view value, Rect bounds, Color color, float scale = 1.5f);
    void button(std::string_view value, Rect bounds, bool selected = false, bool enabled = true, float scale = 1);
    void input(const NEBULA::TextInput& field, Rect bounds, bool focused, bool masked = false, float scale = 1);
    void back();
    void flush(int width, int height);

    GLuint program = 0, vao = 0, vbo = 0;
    GLint canvas = -1;
    std::vector<Vertex> vertices;
};
