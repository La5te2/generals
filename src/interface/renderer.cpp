#include "renderer.hpp"
#include "font.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <string>

namespace {
    #include "icons.inc"

    GLuint compile(GLenum type, const char* source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        GLint ready = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ready);
        if (ready == GL_TRUE) return shader;
        std::array<char, 2048> message{};
        glGetShaderInfoLog(shader, static_cast<GLsizei>(message.size()), nullptr, message.data());
        std::cerr << "Shader compilation failed: " << message.data() << '\n';
        glDeleteShader(shader);
        return 0;
    }
}

Renderer::~Renderer() {
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
}

bool Renderer::init() {
    // convert top-left pixel coordinates to OpenGL's center-origin [-1, 1] coordinates.
    const char* vertexSource = R"(
        #version 330 core
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec4 color;
        uniform vec2 canvas;
        out vec4 tint;
        void main() {
            gl_Position = vec4(position.x / canvas.x * 2.0 - 1.0,
                               1.0 - position.y / canvas.y * 2.0, 0.0, 1.0);
            tint = color;
        }
    )";
    const char* fragmentSource = R"(
        #version 330 core
        in vec4 tint;
        out vec4 pixel;
        void main() { pixel = tint; }
    )";
    GLuint vertex = compile(GL_VERTEX_SHADER, vertexSource);
    GLuint fragment = compile(GL_FRAGMENT_SHADER, fragmentSource);
    if (vertex && fragment) {
        program = glCreateProgram();
        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glLinkProgram(program);
    }
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if (!program) return false;
    GLint ready = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ready);
    if (ready != GL_TRUE) {
        std::array<char, 2048> message{};
        glGetProgramInfoLog(program, static_cast<GLsizei>(message.size()), nullptr, message.data());
        std::cerr << "Shader linking failed: " << message.data() << '\n';
        return false;
    }

    canvas = glGetUniformLocation(program, "canvas");
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // the VAO remembers how position and color are arranged inside each Vertex.
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, position)));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, color)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    vertices.reserve(100000);
    return true;
}

void Renderer::triangle(Point a, Point b, Point c, Color color) {
    vertices.insert(vertices.end(), {{a, color}, {b, color}, {c, color}});
}

void Renderer::rectangle(Rect rect, Color color) {
    Point a{rect.x, rect.y}, b{rect.x + rect.width, rect.y};
    Point c{rect.x + rect.width, rect.y + rect.height}, d{rect.x, rect.y + rect.height};
    triangle(a, b, c, color);
    triangle(a, c, d, color);
}

void Renderer::line(Point a, Point b, float thickness, Color color) {
    float length = std::hypot(b.x - a.x, b.y - a.y);
    if (length == 0) return;
    float dx = (a.y - b.y) * thickness / (2 * length);
    float dy = (b.x - a.x) * thickness / (2 * length);
    Point p{a.x + dx, a.y + dy}, q{b.x + dx, b.y + dy};
    Point r{b.x - dx, b.y - dy}, s{a.x - dx, a.y - dy};
    triangle(p, q, r, color);
    triangle(p, r, s, color);
}

void Renderer::text(std::string_view value, float x, float y, float scale, Color color) {
    for (char letter : value) {
        auto rows = PixelFont::glyph(letter);
        for (int row = 0; row < PixelFont::height; ++row) {
            // adjacent filled pixels share a rectangle, keeping thick strokes continuous.
            int col = 0;
            while (col < PixelFont::width) {
                if ((rows[row] & (1 << (PixelFont::width - 1 - col))) == 0) {
                    ++col;
                    continue;
                }
                int start = col++;
                while (col < PixelFont::width && (rows[row] & (1 << (PixelFont::width - 1 - col)))) ++col;
                rectangle({x + start * scale, y + row * scale, (col - start) * scale, scale}, color);
            }
        }
        x += PixelFont::advance * scale;
    }
}

void Renderer::icon(ViewTerrain terrain, Rect bounds, Color color) {
    std::span<const IconTriangle> mesh;
    std::array<unsigned char, 4> limits{};
    switch (terrain) {
        case ViewTerrain::Mountain: mesh = mountainIcon; limits = mountainBounds; break;
        case ViewTerrain::Obstacle: mesh = obstacleIcon; limits = obstacleBounds; break;
        case ViewTerrain::City: mesh = cityIcon; limits = cityBounds; break;
        case ViewTerrain::General: mesh = crownIcon; limits = crownBounds; break;
        default: return;
    }
    // center the visible outline rather than the original image's transparent margins.
    float width = static_cast<float>(limits[2] - limits[0]);
    float height = static_cast<float>(limits[3] - limits[1]);
    float scale = std::min(bounds.width / width, bounds.height / height);
    float left = bounds.x + (bounds.width - width * scale) / 2;
    float top = bounds.y + (bounds.height - height * scale) / 2;
    auto point = [&](unsigned char x, unsigned char y) {
        return Point{left + (x - limits[0]) * scale, top + (y - limits[1]) * scale};
    };
    for (const IconTriangle& face : mesh) {
        triangle(point(face.ax, face.ay), point(face.bx, face.by), point(face.cx, face.cy), color);
    }
}

void Renderer::drawConsole(const NEBULA::Console& console, int width, int height) {
    constexpr float scale = 1.5f;
    constexpr float spacing = PixelFont::advance * scale;
    Color white{.95f, .96f, .97f}, accent{.40f, .82f, .68f};
    float top = height - 80.0f;

    // wrap feedback above the input line, leaving the turn counter visible below it.
    std::vector<std::string_view> lines;
    std::string_view feedback = console.feedback;
    auto capacity = static_cast<std::size_t>(std::max(1.0f, (width - 64.0f) / spacing));
    while (!feedback.empty()) {
        std::size_t length = std::min(capacity, feedback.size());
        if (length < feedback.size()) {
            auto space = feedback.rfind(' ', length);
            if (space != std::string_view::npos && space > 0) length = space;
        }
        lines.push_back(feedback.substr(0, length));
        feedback.remove_prefix(length);
        while (!feedback.empty() && feedback.front() == ' ') feedback.remove_prefix(1);
    }
    if (!lines.empty()) {
        float outputTop = top - 8 - static_cast<float>(lines.size()) * 22;
        rectangle({16, outputTop, width - 32.0f, top - outputTop}, {.06f, .08f, .09f, .96f});
        for (std::size_t row = 0; row < lines.size(); ++row) {
            text(lines[row], 28, outputTop + 8 + static_cast<float>(row) * 22, scale, white);
        }
    }
    rectangle({16, top, width - 32.0f, 36}, {.06f, .08f, .09f});
    rectangle({16, top, 2, 36}, accent);
    text(">", 26, top + 11, scale, accent);

    // show the part of a long command that contains the cursor.
    auto columns = static_cast<std::size_t>(std::max(1.0f, (width - 76.0f) / spacing));
    std::size_t start = console.cursor >= columns ? console.cursor - columns + 1 : 0;
    text(std::string_view(console.input).substr(start, columns), 46, top + 11, scale, white);
    float cursor = 46 + static_cast<float>(console.cursor - start) * spacing;
    rectangle({cursor, top + 9, 1.5f, 18}, accent);
}

void Renderer::draw(const Observation& view, int perspective, int width, int height, bool running, Tool hover,
                    const NEBULA::Console& console) {
    if (width <= 0 || height <= 0) return;
    vertices.clear();
    glClearColor(.12f, .14f, .15f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    const Color white{.95f, .96f, .97f}, ink{.12f, .14f, .16f};
    const Color red{.82f, .23f, .26f}, blue{.20f, .38f, .73f};
    rectangle({0, 0, static_cast<float>(width), 52}, {.09f, .10f, .11f});
    const std::array<std::string_view, 3> labels{"RED", "BLUE", "ALL"};
    for (int mode = 0; mode < 3; ++mode) {
        Rect button = viewButton(mode);
        Color color = mode == 0 ? red : mode == 1 ? blue : Color{.30f, .34f, .36f};
        rectangle(button, perspective == mode ? color : Color{.16f, .18f, .20f});
        if (perspective == mode) rectangle({button.x, button.y + 26, button.width, 2}, white);
        float length = PixelFont::measure(labels[mode]);
        text(labels[mode], button.x + (button.width - length * 1.5f) / 2,
             button.y + (button.height - PixelFont::height * 1.5f) / 2, 1.5f, white);
    }
    bool ongoing = view.result == Phases::Ongoing;
    for (Tool tool : {Tool::Playback, Tool::Step, Tool::Reset}) {
        if (hover == tool) rectangle(toolButton(tool, width), {.24f, .28f, .30f});
    }
    Color enabled = ongoing ? white : Color{.40f, .43f, .45f};
    Rect playback = toolButton(Tool::Playback, width);
    if (running) {
        rectangle({playback.x + 7, playback.y + 6, 5, 16}, enabled);
        rectangle({playback.x + 16, playback.y + 6, 5, 16}, enabled);
    } else {
        triangle({playback.x + 9, playback.y + 5}, {playback.x + 9, playback.y + 23},
                 {playback.x + 22, playback.y + 14}, enabled);
    }
    Rect single = toolButton(Tool::Step, width);
    Color stepColor = ongoing && !running ? white : Color{.40f, .43f, .45f};
    triangle({single.x + 5, single.y + 6}, {single.x + 5, single.y + 22},
             {single.x + 17, single.y + 14}, stepColor);
    rectangle({single.x + 19, single.y + 6, 3, 16}, stepColor);
    Rect reset = toolButton(Tool::Reset, width);
    Point center{reset.x + 14, reset.y + 14};
    for (int section = 0; section < 16; ++section) {
        float a = .5f + section * .30f, b = a + .30f;
        line({center.x + 8 * std::cos(a), center.y + 8 * std::sin(a)},
             {center.x + 8 * std::cos(b), center.y + 8 * std::sin(b)}, 2, white);
    }
    triangle({center.x + 1, center.y - 12}, {center.x + 7, center.y - 8}, {center.x, center.y - 5}, white);

    if (view.rows > 0 && view.cols > 0 && height > 100 && width > 40) {
        float size = std::min((width - 32.0f) / view.cols, (height - 100.0f) / view.rows);
        float left = (width - size * view.cols) / 2;
        float top = 60 + (height - 100.0f - size * view.rows) / 2;
        for (int row = 0; row < view.rows; ++row) {
            for (int col = 0; col < view.cols; ++col) {
                const ViewCell& cell = view.cells[row * view.cols + col];
                Color fill{.88f, .89f, .88f};
                if (cell.terrain == ViewTerrain::Fog) fill = {.27f, .29f, .31f};
                else if (cell.terrain == ViewTerrain::Obstacle) fill = {.43f, .46f, .48f};
                else if (cell.owner == 0) fill = red;
                else if (cell.owner == 1) fill = blue;
                else if (cell.terrain == ViewTerrain::Mountain) fill = {.70f, .72f, .72f};
                else if (cell.terrain == ViewTerrain::City) fill = {.49f, .52f, .54f};
                Rect tile{left + col * size, top + row * size, size, size};
                rectangle(tile, {.10f, .12f, .13f});
                rectangle({tile.x + .5f, tile.y + .5f, size - 1, size - 1}, fill);
                bool structure = cell.terrain == ViewTerrain::City || cell.terrain == ViewTerrain::General;
                Rect symbol{tile.x + size * .16f, tile.y + size * .16f, size * .68f, size * .68f};
                icon(cell.terrain, symbol, ink);

                // hidden cells carry no army label. zero on an owned or city cell remains meaningful.
                bool visible = cell.terrain != ViewTerrain::Fog && cell.terrain != ViewTerrain::Obstacle;
                if (visible && (cell.owner >= 0 || cell.terrain == ViewTerrain::City || cell.army > 0)) {
                    std::string number = std::to_string(cell.army);
                    float length = PixelFont::measure(number);
                    float scale = std::min({2.5f, size * .048f, size * .84f / length});
                    float x = tile.x + (size - length * scale) / 2;
                    float y = tile.y + (size - PixelFont::height * scale) / 2;
                    // draw the centered number last, with a dark outline over terrain icons.
                    text(number, x - .65f, y, scale, {0, 0, 0, .65f});
                    text(number, x + .65f, y, scale, {0, 0, 0, .65f});
                    text(number, x, y - .65f, scale, {0, 0, 0, .65f});
                    text(number, x, y + .65f, scale, {0, 0, 0, .65f});
                    text(number, x, y, scale, cell.owner >= 0 || structure ? white : ink);
                }
            }
        }
    }
    text("TURN " + std::to_string(view.tick / 2), 16, height - 24.0f, 1.5f, white);
    std::string_view status = running ? "RUNNING" : "PAUSED";
    if (view.result == Phases::RedWin) status = "RED WINS";
    else if (view.result == Phases::BlueWin) status = "BLUE WINS";
    else if (view.result == Phases::Draw) status = "DRAW";
    text(status, width - 16.0f - PixelFont::measure(status) * 1.5f, height - 24.0f, 1.5f, white);
    if (hover != Tool::None) {
        std::string_view label = hover == Tool::Playback ? (running ? "PAUSE" : "PLAY")
                              : hover == Tool::Step ? "HALF TURN" : "NEW MAP";
        Rect button = toolButton(hover, width);
        float length = PixelFont::measure(label) * 1.5f;
        float left = std::min(button.x, width - length - 32.0f);
        rectangle({left, 48, length + 16, 28}, {.06f, .07f, .08f});
        text(label, left + 8, 48 + (28 - PixelFont::height * 1.5f) / 2, 1.5f, white);
    }

    if (console.opened) drawConsole(console, width, height);

    // all rectangles, glyph pixels and icons share one vertex buffer and one draw call.
    glUseProgram(program);
    glUniform2f(canvas, static_cast<float>(width), static_cast<float>(height));
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
}
