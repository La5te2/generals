// presentation layer: turn game observations and interface state into the board, controls and text shown in the window.
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

    // compile one GLSL stage and return its OpenGL handle. reports a compilation failure if return value is 0.
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
    // convert top-left window coordinates to [-1, 1], with positive y pointing upward in OpenGL.
    // canvas holds the logical window size. glViewport() maps the result to framebuffer pixels.
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
    // every triangle carries its color through the vertex stage to the fragment stage.
    const char* fragmentSource = R"(
        #version 330 core
        in vec4 tint;
        out vec4 pixel;
        void main() { pixel = tint; }
    )";
    GLuint vertex = compile(GL_VERTEX_SHADER, vertexSource);
    GLuint fragment = compile(GL_FRAGMENT_SHADER, fragmentSource);
    // link both stages into one program, matching the vertex output tint to the fragment input tint.
    if (vertex && fragment) {
        program = glCreateProgram();
        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glLinkProgram(program);
    }
    // the linked program retains its compiled stages after these shader handles are released.
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
    // the VBO stores vertex data on the GPU. the VAO stores how to read that data.
    // locations 0 and 1 match the shader inputs, and each vertex occupies sizeof(Vertex) bytes.
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, position)));
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          reinterpret_cast<void*>(offsetof(Vertex, color)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    // alpha blends translucent panels and text outlines with the colors already drawn behind them.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    vertices.reserve(100000);
    return true;
}

// accumulate geometry in CPU memory. flush() uploads all triangles after the page is assembled.
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
    // offset both endpoints perpendicular to the segment by half its thickness, forming a rectangle.
    float dx = (a.y - b.y) * thickness / (2 * length);
    float dy = (b.x - a.x) * thickness / (2 * length);
    Point p{a.x + dx, a.y + dy}, q{b.x + dx, b.y + dy};
    Point r{b.x - dx, b.y - dy}, s{a.x - dx, a.y - dy};
    triangle(p, q, r, color);
    triangle(p, r, s, color);
}

// glyph rectangles are prepared in font.hpp. drawing only applies position, scale and color.
void Renderer::text(std::string_view value, float x, float y, float scale, Color color) {
    for (char letter : value) {
        for (const auto& stroke : PixelFont::strokes(letter)) {
            rectangle({x + stroke.col * scale, y + stroke.row * scale, stroke.length * scale, scale}, color);
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
    // fit the icon's visible bounds inside the destination rectangle, preserving its proportions.
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

void Renderer::drawConsole(const NEBULA::Console& console, int width, int height, float textScale) {
    float scale = 1.5f * textScale;
    float spacing = PixelFont::advance * scale;
    Color white{.95f, .96f, .97f}, accent{.40f, .82f, .68f};
    float top = height - 44 - 36 * textScale;

    // wrap feedback above the input line, leaving the turn counter visible below it.
    std::vector<std::string_view> lines;
    std::string_view feedback = console.feedback;
    auto capacity = static_cast<std::size_t>(std::max(1.0f, (width - 32 - 32 * textScale) / spacing));
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
        float outputTop = top - (8 + static_cast<float>(lines.size()) * 22) * textScale;
        rectangle({16, outputTop, width - 32.0f, top - outputTop}, {.06f, .08f, .09f, .96f});
        for (std::size_t row = 0; row < lines.size(); ++row) {
            text(lines[row], 16 + 12 * textScale, outputTop + (8 + static_cast<float>(row) * 22) * textScale, scale, white);
        }
    }
    rectangle({16, top, width - 32.0f, 36 * textScale}, {.06f, .08f, .09f});
    rectangle({16, top, 2, 36 * textScale}, accent);
    text(">", 16 + 10 * textScale, top + 11 * textScale, scale, accent);

    // show the part of a long command that contains the cursor.
    auto columns = static_cast<std::size_t>(std::max(1.0f, (width - 32 - 44 * textScale) / spacing));
    std::size_t start = console.cursor >= columns ? console.cursor - columns + 1 : 0;
    float left = 16 + 30 * textScale;
    text(std::string_view(console.input).substr(start, columns), left, top + 11 * textScale, scale, white);
    float cursor = left + static_cast<float>(console.cursor - start) * spacing;
    rectangle({cursor, top + 9 * textScale, 1.5f * textScale, 18 * textScale}, accent);
}

void Renderer::label(std::string_view value, Rect bounds, Color color, float scale) {
    // center the text and reduce its requested scale only when the label exceeds the available width.
    float length = PixelFont::measure(value);
    if (length > 0) scale = std::min(scale, (bounds.width - 12) / length);
    text(value, bounds.x + (bounds.width - length * scale) / 2,
         bounds.y + (bounds.height - PixelFont::height * scale) / 2, scale, color);
}

void Renderer::button(std::string_view value, Rect bounds, bool selected, bool enabled, float scale) {
    rectangle(bounds, selected ? Color{.16f, .40f, .34f} : Color{.20f, .23f, .25f});
    if (selected) rectangle({bounds.x, bounds.y + bounds.height - 2 * scale, bounds.width, 2 * scale}, {.40f, .82f, .68f});
    label(value, bounds, enabled ? Color{.95f, .96f, .97f} : Color{.43f, .46f, .48f}, 1.5f * scale);
}

void Renderer::input(const NEBULA::TextInput& field, Rect bounds, bool focused, bool masked, float scale) {
    rectangle(bounds, {.06f, .08f, .09f});
    if (focused) rectangle({bounds.x, bounds.y + bounds.height - 2 * scale, bounds.width, 2 * scale}, {.40f, .82f, .68f});
    // reserve space for the cursor and scroll long input horizontally to keep that cursor visible.
    auto columns = static_cast<std::size_t>(std::max(1.0f, (bounds.width - 24 * scale) / (12 * scale)));
    std::size_t start = field.cursor >= columns ? field.cursor - columns + 1 : 0;
    // masking changes the displayed copy. the input field retains the original text.
    std::string shown = masked ? std::string(field.input.size(), '*') : field.input;
    text(std::string_view(shown).substr(start, columns), bounds.x + 10 * scale, bounds.y + 11 * scale,
         1.5f * scale, {.95f, .96f, .97f});
    if (focused) rectangle({bounds.x + (10 + static_cast<float>(field.cursor - start) * 12) * scale,
                            bounds.y + 9 * scale, 1.5f * scale, 18 * scale}, {.40f, .82f, .68f});
}

void Renderer::back(float scale) {
    Rect bounds = NEBULA::backButton(scale);
    Color white{.95f, .96f, .97f};
    line({bounds.x + 6 * scale, bounds.y + 14 * scale}, {bounds.x + 23 * scale, bounds.y + 14 * scale}, 2 * scale, white);
    line({bounds.x + 6 * scale, bounds.y + 14 * scale}, {bounds.x + 13 * scale, bounds.y + 7 * scale}, 2 * scale, white);
    line({bounds.x + 6 * scale, bounds.y + 14 * scale}, {bounds.x + 13 * scale, bounds.y + 21 * scale}, 2 * scale, white);
}

void Renderer::drawTools(const NEBULA::BoardControls& controls, int width, float scale) {
    for (Tool tool : NEBULA::tools) {
        Rect bounds = toolButton(tool, width, scale);
        bool enabled = controls.enabled(tool);
        Color color = enabled ? Color{.95f, .96f, .97f} : Color{.36f, .39f, .41f};
        if (enabled && controls.hover == tool) rectangle(bounds, {.24f, .28f, .30f});
        auto point = [&](float x, float y) { return Point{bounds.x + x * scale, bounds.y + y * scale}; };
        if (tool == Tool::Playback) {
            if (controls.running) {
                rectangle({bounds.x + 7 * scale, bounds.y + 6 * scale, 5 * scale, 16 * scale}, color);
                rectangle({bounds.x + 16 * scale, bounds.y + 6 * scale, 5 * scale, 16 * scale}, color);
            } else triangle(point(9, 5), point(9, 23), point(22, 14), color);
        } else if (tool == Tool::Backward || tool == Tool::Forward) {
            // mirror the forward-step symbol for backward stepping.
            auto x = [&](float value) { return tool == Tool::Backward ? 28 - value : value; };
            triangle(point(x(5), 6), point(x(5), 22), point(x(17), 14), color);
            line(point(x(20.5f), 6), point(x(20.5f), 22), 3 * scale, color);
        } else if (tool == Tool::Stop) {
            rectangle({bounds.x + 7 * scale, bounds.y + 7 * scale, 14 * scale, 14 * scale}, color);
        } else if (tool == Tool::Reset) {
            constexpr float pi = 3.14159265f;
            Point last = point(14 + 8 * std::cos(pi / 4), 14 + 8 * std::sin(pi / 4));
            for (int segment = 1; segment <= 24; ++segment) {
                float angle = pi / 4 + 1.5f * pi * segment / 24;
                Point next = point(14 + 8 * std::cos(angle), 14 + 8 * std::sin(angle));
                line(last, next, 2 * scale, color);
                last = next;
            }
            triangle(point(23, 12), point(16, 10), point(22, 5), color);
        }
    }
}

// assemble the selected configuration page, then submit it to the back buffer in one batch.
void Renderer::drawSetup(const NEBULA::Setup& setup, int width, int height, const NEBULA::Console& console) {
    using namespace NEBULA;
    if (width <= 0 || height <= 0) return;
    // rebuild the complete frame, including its background, from the current page state.
    vertices.clear();
    glClearColor(.12f, .14f, .15f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    Color white{.95f, .96f, .97f}, muted{.62f, .66f, .68f};
    float scale = contentScale(width, height);
    float textScale = barScale(height);
    if (setup.scene == Scene::Home) {
        label("PLAY", menuTitle(width, height), white, 3 * scale);
        const std::array<std::string_view, 3> names{"LOCAL", "ONLINE", "REPLAY"};
        const std::array<ViewTerrain, 3> symbols{ViewTerrain::General, ViewTerrain::City, ViewTerrain::Mountain};
        for (int index = 0; index < 3; ++index) {
            Rect bounds = menuButton(index, width, height);
            rectangle(bounds, white);
            label(names[index], bounds, {.12f, .14f, .15f}, 1.5f * scale);
            icon(symbols[index], {bounds.x + 16 * scale, bounds.y + 12 * scale, 28 * scale, 28 * scale}, index == 0
                 ? Color{.72f, .23f, .26f} : index == 1 ? Color{.16f, .40f, .34f} : Color{.55f, .44f, .16f});
        }
    } else {
        // the top bar has a tighter scale range than the central form.
        back(textScale);
        std::string_view title = setup.scene == Scene::Local ? "LOCAL" : setup.scene == Scene::Online ? "ONLINE" : "REPLAY";
        float font = 1.5f * textScale;
        float baseline = 12 + (28 * textScale - PixelFont::height * font) / 2;
        text(title, 16 + 48 * textScale, baseline, font, white);

        Rect start = startButton(width, textScale);
        Color action = setup.scene != Scene::Online ? white : muted;
        std::string_view command = setup.scene == Scene::Local ? "START MATCH" : setup.scene == Scene::Online ? "CONNECT" : "PLAY";
        text(command, start.x + start.width - 36 * textScale - PixelFont::measure(command) * font, baseline, font, action);
        float right = start.x + start.width - 5 * textScale, middle = start.y + start.height / 2;
        line({right - 18 * textScale, middle}, {right, middle}, 2 * textScale, action);
        line({right - 7 * textScale, middle - 7 * textScale}, {right, middle}, 2 * textScale, action);
        line({right - 7 * textScale, middle + 7 * textScale}, {right, middle}, 2 * textScale, action);

        // scene.hpp provides the same control rectangles to drawing and mouse hit testing.
        auto group = [&](std::string_view name, int index) {
            Rect row = formControl(setup.scene, index, width, height);
            text(name, row.x, row.y - 24 * scale, 1.5f * scale, muted);
            return row;
        };
        auto fileInput = [&](std::string_view name, int field, int index) {
            group(name, index);
            Rect entry = inputField(setup.scene, field, width, height);
            input(setup.fields[field], entry, setup.focus == field, false, scale);
            if (setup.focus != field && setup.humanPlayer(field)) {
                text("HUMAN", entry.x + 10 * scale, entry.y + 11 * scale, 1.5f * scale, muted);
            }
            Rect bounds = fileButton(setup.scene, field, width, height);
            rectangle(bounds, {.06f, .08f, .09f});
            Color symbol = setup.fileHover == field ? Color{.40f, .82f, .68f} : white;
            // the folder outline uses the same scaled coordinates as the surrounding controls.
            std::array<Point, 7> outline{{{8, 10}, {15, 10}, {18, 14}, {28, 14}, {28, 26}, {8, 26}, {8, 10}}};
            for (std::size_t point = 1; point < outline.size(); ++point) {
                line({bounds.x + outline[point - 1].x * scale, bounds.y + outline[point - 1].y * scale},
                     {bounds.x + outline[point].x * scale, bounds.y + outline[point].y * scale}, 1.5f * scale, symbol);
            }
            if (setup.fileHover == field && !console.opened) {
                Rect tooltip{bounds.x + bounds.width - 120 * scale, bounds.y - 30 * scale, 120 * scale, 24 * scale};
                rectangle(tooltip, {.06f, .07f, .08f});
                label(field == 6 ? "DIRECTORY" : "SELECT FILE", tooltip, white, 1.25f * scale);
            }
        };
        if (setup.scene == Scene::Local) {
            fileInput("RED PLAYER", 3, 0);
            fileInput("BLUE PLAYER", 4, 1);
            fileInput("DIRECTORY", 6, 2);
        } else if (setup.scene == Scene::Online) {
            Rect server = group("SERVER", 0);
            button("BOT", choiceButton(server, 0, 2), !setup.mainServer, true, scale);
            button("MAIN", choiceButton(server, 1, 2), setup.mainServer, true, scale);
            fileInput("PLAYER", 5, 1);
            input(setup.fields[0], group("USERNAME", 2), setup.focus == 0, false, scale);
            input(setup.fields[1], group("USER ID", 3), setup.focus == 1, true, scale);
        } else {
            fileInput("REPLAY FILE", 2, 0);
        }
    }
    drawMessage(setup, width, height);
    if (console.opened) drawConsole(console, width, height, textScale);
    flush(width, height);
}

void Renderer::drawMessage(const NEBULA::Setup& setup, int width, int height, bool board) {
    // operation feedback wraps at word boundaries and fades with the notification's elapsed time.
    float textScale = NEBULA::barScale(height);
    std::string_view message = setup.message;
    float opacity = setup.messageOpacity();
    Color feedback{.62f, .66f, .68f, opacity};
    Rect row = NEBULA::messageArea(width, textScale);
    if (board) {
        Rect names = NEBULA::scoreNamesArea(width, height);
        row = {names.x, names.y + names.height + 44 * textScale, names.width, row.height};
    }
    auto columns = static_cast<std::size_t>(std::max(1.0f, row.width / (10 * textScale)));
    float top = row.y;
    while (!message.empty()) {
        std::size_t length = std::min(columns, message.size());
        if (length < message.size()) {
            auto space = message.rfind(' ', length);
            if (space != std::string_view::npos && space > 0) length = space;
        }
        text(message.substr(0, length), row.x, top, 1.25f * textScale, feedback);
        message.remove_prefix(length);
        while (!message.empty() && message.front() == ' ') message.remove_prefix(1);
        top += 18 * textScale;
    }
}

void Renderer::drawScores(const Observation& view, const NEBULA::Setup& setup, int width, int height) {
    Rect area = NEBULA::scoreArea(width, height);
    float scale = NEBULA::barScale(height);
    std::array<std::string_view, 2> names{"RED", "BLUE"};
    if (setup.scene == NEBULA::Scene::Online && view.player >= 0 && view.player < 2) {
        if (!setup.fields[0].input.empty()) names[view.player] = setup.fields[0].input;
        names[1 - view.player] = "OPPONENT";
    }
    const std::array<Color, 2> colors{{{.82f, .23f, .26f}, {.20f, .38f, .73f}}};
    const Color paper{.95f, .96f, .97f}, ink{.08f, .10f, .11f};
    const std::array<std::string_view, 3> headings{"", "Army", "Land"};
    constexpr std::array<float, 4> columns{0, .16f, .58f, 1};
    float border = scale, rowHeight = area.height / 3;
    rectangle(area, ink);
    // the first column shows the player's color. the two score columns keep their headings and numbers centered.
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            Rect cell{area.x + area.width * columns[column] + border, area.y + row * rowHeight + border,
                      area.width * (columns[column + 1] - columns[column]) - border * 2, rowHeight - border * 2};
            bool playerCell = row > 0 && column == 0;
            rectangle(cell, playerCell ? colors[row - 1] : paper);
            std::string value;
            if (row == 0) value = headings[column];
            else if (column > 0) value = std::to_string(column == 1 ? view.armies[row - 1] : view.land[row - 1]);
            label(value, cell, playerCell ? paper : ink, 1.25f * scale);
        }
    }
    Rect players = NEBULA::scoreNamesArea(width, height);
    for (int player = 0; player < 2; ++player) {
        Rect row{players.x, players.y + player * players.height / 2, players.width, players.height / 2};
        rectangle({row.x, row.y + (row.height - 8 * scale) / 2, 8 * scale, 8 * scale}, colors[player]);
        row.x += 14 * scale;
        row.width -= 14 * scale;
        label(names[player], row, paper, scale);
    }
}

// draw the supplied observation. selecting and updating that observation belongs to WindowState.
void Renderer::draw(const Observation& view, int perspective, int width, int height,
                    const NEBULA::BoardControls& controls, const NEBULA::Setup& setup,
                    const NEBULA::Console& console, const NEBULA::Controller& controller, std::span<const Action> queued) {
    if (width <= 0 || height <= 0) return;
    vertices.clear();
    glClearColor(.12f, .14f, .15f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    const Color white{.95f, .96f, .97f}, ink{.12f, .14f, .16f};
    const Color red{.82f, .23f, .26f}, blue{.20f, .38f, .73f};
    const Color darkRed{.53f, .12f, .15f}, darkBlue{.11f, .23f, .48f};
    float textScale = NEBULA::barScale(height);
    back(textScale);
    float font = 1.5f * textScale;
    bool human = controls.human, active = controls.active, running = controls.running;
    const std::array<std::string_view, 3> labels{"RED", "BLUE", "ALL"};
    for (int mode = 0; controls.spectator() && mode < 3; ++mode) {
        Rect button = viewButton(mode, textScale);
        Color color = mode == 0 ? red : mode == 1 ? blue : Color{.30f, .34f, .36f};
        rectangle(button, perspective == mode ? color : Color{.16f, .18f, .20f});
        if (perspective == mode) rectangle({button.x, button.y + 26 * textScale, button.width, 2 * textScale}, white);
        float length = PixelFont::measure(labels[mode]);
        text(labels[mode], button.x + (button.width - length * font) / 2,
             button.y + (button.height - PixelFont::height * font) / 2, font, white);
    }
    drawScores(view, setup, width, height);
    drawTools(controls, width, textScale);

    Rect board = NEBULA::boardArea(view.rows, view.cols, width, height);
    if (board.width > 0) {
        float size = board.width / view.cols;
        float left = board.x, top = board.y;
        for (int row = 0; row < view.rows; ++row) {
            for (int col = 0; col < view.cols; ++col) {
                const ViewCell& cell = view.cells[row * view.cols + col];
                bool structure = cell.terrain == ViewTerrain::City || cell.terrain == ViewTerrain::General;
                Color fill{.88f, .89f, .88f};
                if (cell.terrain == ViewTerrain::Fog) fill = {.27f, .29f, .31f};
                else if (cell.terrain == ViewTerrain::Obstacle) fill = {.43f, .46f, .48f};
                else if (cell.owner == 0) fill = structure ? darkRed : red;
                else if (cell.owner == 1) fill = structure ? darkBlue : blue;
                else if (cell.terrain == ViewTerrain::Mountain) fill = {.70f, .72f, .72f};
                else if (cell.terrain == ViewTerrain::City) fill = {.49f, .52f, .54f};
                // an inset fill exposes the dark tile beneath as the grid border.
                Rect tile{left + col * size, top + row * size, size, size};
                rectangle(tile, {.10f, .12f, .13f});
                rectangle({tile.x + .5f, tile.y + .5f, size - 1, size - 1}, fill);
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
        if (human) drawSelection(controller, queued, board, view.cols);
    }
    // two half-turns form one displayed turn. bottom text shares the top bar's limited scaling.
    float bottom = height - 10.5f - PixelFont::height * font;
    text("TURN " + std::to_string(view.tick / 2), 16, bottom, font, white);
    if (human && controller.half()) {
        label("50%", {width / 2.0f - 40 * textScale, bottom - 4, 80 * textScale, 24 * textScale},
              {.98f, .80f, .32f}, font);
    }
    std::string_view status = active ? (running ? "RUNNING" : "PAUSED") : "STOPPED";
    if (view.result == Phases::RedWin) status = "RED WINS";
    else if (view.result == Phases::BlueWin) status = "BLUE WINS";
    else if (view.result == Phases::Draw) status = "DRAW";
    text(status, width - 16.0f - PixelFont::measure(status) * font, bottom, font, white);
    drawMessage(setup, width, height, true);
    if (controls.hover != Tool::None && !console.opened) {
        const std::array<std::string_view, 6> names{"", "STEP BACK", running ? "PAUSE" : "PLAY", "STEP FORWARD", "STOP", "RESET"};
        std::string_view label = names[static_cast<int>(controls.hover)];
        Rect button = toolButton(controls.hover, width, textScale);
        float length = PixelFont::measure(label) * font;
        float left = std::min(button.x, width - length - 32.0f);
        Rect players = NEBULA::scoreNamesArea(width, height);
        float top = players.y + players.height + 8;
        rectangle({left, top, length + 16, 28 * textScale}, {.06f, .07f, .08f});
        text(label, left + 8, top + (28 * textScale - PixelFont::height * font) / 2, font,
             controls.enabled(controls.hover) ? white : Color{.62f, .66f, .68f});
    }

    if (console.opened) drawConsole(console, width, height, textScale);
    flush(width, height);
}

void Renderer::drawSelection(const NEBULA::Controller& controller, std::span<const Action> queued, Rect board, int cols) {
    float size = board.width / cols;
    Color white{.97f, .98f, .99f}, gold{.98f, .80f, .32f}, outline{.06f, .08f, .09f};
    // place queue arrows across tile edges, leaving the centered army numbers visible.
    for (const Action& action : queued) {
        float dx = 0, dy = 0;
        switch (action.direction) {
            case Direction::Up: dy = -1; break;
            case Direction::Down: dy = 1; break;
            case Direction::Left: dx = -1; break;
            case Direction::Right: dx = 1; break;
        }
        Point edge{board.x + (action.col + .5f + dx * .5f) * size,
                   board.y + (action.row + .5f + dy * .5f) * size};
        Point from{edge.x - dx * size * .15f, edge.y - dy * size * .15f};
        Point to{edge.x + dx * size * .15f, edge.y + dy * size * .15f};
        for (int layer = 0; layer < 2; ++layer) {
            float thickness = size * (layer == 0 ? .11f : .055f);
            Color color = layer == 0 ? outline : action.half ? gold : white;
            line(from, to, thickness, color);
            line({edge.x - dy * size * .12f, edge.y + dx * size * .12f}, to, thickness, color);
            line({edge.x + dy * size * .12f, edge.y - dx * size * .12f}, to, thickness, color);
        }
    }
    int cell = controller.selection();
    if (cell < 0) return;
    float inset = size * .06f, thickness = size * .055f;
    float left = board.x + (cell % cols) * size + inset, top = board.y + (cell / cols) * size + inset;
    float span = size - inset * 2;
    Color color = controller.half() ? gold : white;
    rectangle({left, top, span, thickness}, color);
    rectangle({left, top + span - thickness, span, thickness}, color);
    rectangle({left, top, thickness, span}, color);
    rectangle({left + span - thickness, top, thickness, span}, color);
}

void Renderer::flush(int width, int height) {
    // upload the CPU vertex list to the VBO and draw its triangles in their insertion order.
    // later shapes cover earlier ones, placing text over icons and the console over the page.
    glUseProgram(program);
    glUniform2f(canvas, static_cast<float>(width), static_cast<float>(height));
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(), GL_STREAM_DRAW);
    // this submits drawing commands. drawWindow() in nebula.cpp presents the frame with glfwSwapBuffers().
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
}
