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

// each glyph is a bit grid from font.hpp. filled bits become rectangles at the requested scale.
void Renderer::text(std::string_view value, float x, float y, float scale, Color color) {
    for (char letter : value) {
        auto rows = PixelFont::glyph(letter);
        for (int row = 0; row < static_cast<int>(rows.size()); ++row) {
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

void Renderer::back() {
    Rect bounds = NEBULA::backButton();
    Color white{.95f, .96f, .97f};
    line({bounds.x + 6, bounds.y + 14}, {bounds.x + 23, bounds.y + 14}, 2, white);
    line({bounds.x + 6, bounds.y + 14}, {bounds.x + 13, bounds.y + 7}, 2, white);
    line({bounds.x + 6, bounds.y + 14}, {bounds.x + 13, bounds.y + 21}, 2, white);
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
        // the top bar uses fixed sizes. only the central form below uses contentScale().
        rectangle({0, 0, static_cast<float>(width), 52}, {.09f, .10f, .11f});
        back();
        std::string_view title = setup.scene == Scene::Local ? "LOCAL" : setup.scene == Scene::Online ? "ONLINE" : "REPLAY";
        text(title, 64, 19, 1.5f, white);

        Rect start = startButton(width);
        Color action = setup.scene == Scene::Local ? white : muted;
        std::string_view command = setup.scene == Scene::Local ? "START MATCH" : setup.scene == Scene::Online ? "CONNECT" : "PLAY";
        text(command, start.x + start.width - 36 - PixelFont::measure(command) * 1.5f, 19, 1.5f, action);
        float right = start.x + start.width - 5, middle = start.y + start.height / 2;
        line({right - 18, middle}, {right, middle}, 2, action);
        line({right - 7, middle - 7}, {right, middle}, 2, action);
        line({right - 7, middle + 7}, {right, middle}, 2, action);

        // scene.hpp provides the same control rectangles to drawing and mouse hit testing.
        auto group = [&](std::string_view name, int index) {
            Rect row = formControl(setup.scene, index, width, height);
            text(name, row.x, row.y - 24 * scale, 1.5f * scale, muted);
            return row;
        };
        auto fileInput = [&](std::string_view name, int field, int index) {
            group(name, index);
            input(setup.fields[field], inputField(setup.scene, field, width, height), setup.focus == field, false, scale);
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
                label("SELECT FILE", tooltip, white, 1.25f * scale);
            }
        };
        if (setup.scene == Scene::Local) {
            fileInput("RED PLAYER", 3, 0);
            fileInput("BLUE PLAYER", 4, 1);
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
    // show operation feedback below the top bar at a fixed text size, wrapping at word boundaries.
    std::string_view message = setup.message;
    Color feedback{muted.r, muted.g, muted.b, setup.messageOpacity()};
    Rect row = messageArea(width);
    auto columns = static_cast<std::size_t>(std::max(1.0f, row.width / 10));
    float top = row.y;
    while (!message.empty()) {
        std::size_t length = std::min(columns, message.size());
        if (length < message.size()) {
            auto space = message.rfind(' ', length);
            if (space != std::string_view::npos && space > 0) length = space;
        }
        text(message.substr(0, length), row.x, top, 1.25f, feedback);
        message.remove_prefix(length);
        while (!message.empty() && message.front() == ' ') message.remove_prefix(1);
        top += 18;
    }
    if (console.opened) drawConsole(console, width, height);
    flush(width, height);
}

// draw the supplied observation. selecting and updating that observation belongs to WindowState.
void Renderer::draw(const Observation& view, int perspective, int width, int height, bool running, bool active, Tool hover,
                    const NEBULA::Console& console) {
    if (width <= 0 || height <= 0) return;
    vertices.clear();
    glClearColor(.12f, .14f, .15f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    const Color white{.95f, .96f, .97f}, ink{.12f, .14f, .16f};
    const Color red{.82f, .23f, .26f}, blue{.20f, .38f, .73f};
    const Color darkRed{.53f, .12f, .15f}, darkBlue{.11f, .23f, .48f};
    rectangle({0, 0, static_cast<float>(width), 52}, {.09f, .10f, .11f});
    back();
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
    if (active) {
        for (Tool tool : {Tool::Playback, Tool::Step, Tool::Stop}) {
            if (hover == tool) rectangle(toolButton(tool, width), {.24f, .28f, .30f});
        }
        Rect playback = toolButton(Tool::Playback, width);
        if (running) {
            rectangle({playback.x + 7, playback.y + 6, 5, 16}, white);
            rectangle({playback.x + 16, playback.y + 6, 5, 16}, white);
        } else {
            triangle({playback.x + 9, playback.y + 5}, {playback.x + 9, playback.y + 23},
                     {playback.x + 22, playback.y + 14}, white);
        }
        Rect single = toolButton(Tool::Step, width);
        Color stepColor = !running ? white : Color{.40f, .43f, .45f};
        triangle({single.x + 5, single.y + 6}, {single.x + 5, single.y + 22},
                 {single.x + 17, single.y + 14}, stepColor);
        rectangle({single.x + 19, single.y + 6, 3, 16}, stepColor);
        Rect stop = toolButton(Tool::Stop, width);
        rectangle({stop.x + 7, stop.y + 7, 14, 14}, white);
    } else button("SETUP", NEBULA::setupButton(width));

    if (view.rows > 0 && view.cols > 0 && height > 100 && width > 40) {
        // choose the largest square cell that fits the available width and height between the bars.
        float size = std::min((width - 32.0f) / view.cols, (height - 100.0f) / view.rows);
        float left = (width - size * view.cols) / 2;
        float top = 60 + (height - 100.0f - size * view.rows) / 2;
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
    }
    // two half-turns form one displayed turn. the bottom labels keep their size as the board scales.
    text("TURN " + std::to_string(view.tick / 2), 16, height - 24.0f, 1.5f, white);
    std::string_view status = active ? (running ? "RUNNING" : "PAUSED") : "STOPPED";
    if (view.result == Phases::RedWin) status = "RED WINS";
    else if (view.result == Phases::BlueWin) status = "BLUE WINS";
    else if (view.result == Phases::Draw) status = "DRAW";
    text(status, width - 16.0f - PixelFont::measure(status) * 1.5f, height - 24.0f, 1.5f, white);
    if (active && hover != Tool::None) {
        std::string_view label = hover == Tool::Playback ? (running ? "PAUSE" : "PLAY")
                              : hover == Tool::Step ? "HALF TURN" : "STOP";
        Rect button = toolButton(hover, width);
        float length = PixelFont::measure(label) * 1.5f;
        float left = std::min(button.x, width - length - 32.0f);
        rectangle({left, 48, length + 16, 28}, {.06f, .07f, .08f});
        text(label, left + 8, 48 + (28 - PixelFont::height * 1.5f) / 2, 1.5f, white);
    }

    if (console.opened) drawConsole(console, width, height);
    flush(width, height);
}

void Renderer::flush(int width, int height) {
    // upload the CPU vertex list to the VBO and draw its triangles in their insertion order.
    // later shapes cover earlier ones, placing text over icons and the console over the page.
    glUseProgram(program);
    glUniform2f(canvas, static_cast<float>(width), static_cast<float>(height));
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(), GL_STREAM_DRAW);
    // this submits drawing commands. redraw() in nebula.cpp presents the frame with glfwSwapBuffers().
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
}
