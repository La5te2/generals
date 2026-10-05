#include "engine/engine.hpp"
#include "clock.hpp"
#include "renderer.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <iostream>
#include <random>

namespace NEBULA {
    struct WindowState {
        Engine engine;
        Renderer renderer;
        Clock clock;
        Console console;
        Observation view;
        bool closing = false;
        int perspective = 2; // 0 = red, 1 = blue, 2 = full board.
        Tool hover = Tool::None;

        void update() {
            view = *engine.observe(perspective == 1 ? 1 : 0);
            if (perspective != 2) return;
            // the full board is a viewer option. strategy input still comes from engine.observe().
            auto state = engine.snapshot();
            for (int row = 0; row < view.rows; ++row) {
                for (int col = 0; col < view.cols; ++col) {
                    const Cell& cell = state->board.at(row, col);
                    ViewCell& shown = view.cells[row * view.cols + col];
                    switch (cell.terrain) {
                        case Terrain::Plain: shown.terrain = ViewTerrain::Plain; break;
                        case Terrain::Mountain: shown.terrain = ViewTerrain::Mountain; break;
                        case Terrain::City: shown.terrain = ViewTerrain::City; break;
                        case Terrain::General: shown.terrain = ViewTerrain::General; break;
                    }
                    shown.owner = cell.owner;
                    shown.army = cell.army;
                }
            }
        }

        bool regenerate() {
            std::random_device random;
            if (engine.reset(0, 0, random())) {
                clock.reset();
                update();
                return true;
            }
            std::cerr << "Map generation failed\n";
            return false;
        }

        void advance() {
            // both sides pass until human and strategy inputs supply their actions here.
            if (engine.step({Action{}, Action{}})) update();
            if (view.result != Phases::Ongoing) clock.pause();
        }

        void activate(Tool tool) {
            if (tool == Tool::Reset) regenerate();
            else if (view.result == Phases::Ongoing) {
                if (tool == Tool::Playback) {
                    if (clock.running()) clock.pause();
                    else clock.resume();
                } else if (tool == Tool::Step && !clock.running()) {
                    clock.reset();
                    advance();
                }
            }
        }

        // execute commands against this local game. console.cpp only interprets the input text.
        void runCommand(Command command) {
            if ((command == Command::Pause || command == Command::Resume || command == Command::Step) &&
                view.result != Phases::Ongoing) {
                console.feedback = "Game ended";
                return;
            }
            switch (command) {
                case Command::None: break;
                case Command::Help:
                    console.feedback = "help pause resume step restart quit";
                    break;
                case Command::Pause:
                    clock.pause();
                    console.feedback = "Paused";
                    break;
                case Command::Resume:
                    clock.resume();
                    console.feedback = "Running";
                    break;
                case Command::Step:
                    if (clock.running()) console.feedback = "Pause the game before stepping";
                    else {
                        activate(Tool::Step);
                        console.feedback = view.result == Phases::Ongoing ? "Advanced one half-turn" : "Game ended";
                    }
                    break;
                case Command::Restart:
                    console.feedback = regenerate() ? "New map: paused" : "Map generation failed";
                    break;
                case Command::Quit:
                    closing = true;
                    break;
                case Command::Invalid:
                    console.feedback = "Unknown command or extra arguments";
                    break;
            }
        }
    };

    void redraw(GLFWwindow* window) {
        auto& app = *static_cast<WindowState*>(glfwGetWindowUserPointer(window));
        int width, height, pixelsWide, pixelsHigh;
        glfwGetWindowSize(window, &width, &height);
        glfwGetFramebufferSize(window, &pixelsWide, &pixelsHigh);
        if (pixelsWide == 0 || pixelsHigh == 0) return;
        // window units locate controls. framebuffer pixels set the viewport on high-DPI displays.
        glViewport(0, 0, pixelsWide, pixelsHigh);
        app.renderer.draw(app.view, app.perspective, width, height, app.clock.running(), app.hover, app.console);
        glfwSwapBuffers(window);
    }

    int run(GLFWwindow* window) {
        // app is destroyed when run() returns, while main() still owns the live OpenGL window.
        WindowState app;
        if (!app.renderer.init() || !app.engine.reset()) return 1;
        app.update();
        glfwSetWindowUserPointer(window, &app);
        // Windows can keep glfwWaitEvents() inside its move/resize loop.
        // these callbacks draw immediately while that loop is handling repaint and resize events.
        glfwSetWindowRefreshCallback(window, redraw);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* target, int, int) { redraw(target); });
        glfwSetCursorPosCallback(window, [](GLFWwindow* target, double x, double y) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            if (state.console.opened) return;
            int width, height;
            glfwGetWindowSize(target, &width, &height);
            Tool hover = Tool::None;
            for (Tool tool : {Tool::Playback, Tool::Step, Tool::Reset}) {
                if (toolButton(tool, width).contains(x, y)) hover = tool;
            }
            if (hover != state.hover) {
                state.hover = hover;
                redraw(target);
            }
        });
        glfwSetCursorEnterCallback(window, [](GLFWwindow* target, int entered) {
            if (entered) return;
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            state.hover = Tool::None;
            redraw(target);
        });
        glfwSetMouseButtonCallback(window, [](GLFWwindow* target, int button, int action, int) {
            if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS) return;
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            if (state.console.opened) return;
            double x, y;
            glfwGetCursorPos(target, &x, &y);
            for (int mode = 0; mode < 3; ++mode) {
                if (viewButton(mode).contains(x, y)) {
                    state.perspective = mode;
                    state.update();
                }
            }
            int width, height;
            glfwGetWindowSize(target, &width, &height);
            for (Tool tool : {Tool::Playback, Tool::Step, Tool::Reset}) {
                if (toolButton(tool, width).contains(x, y)) state.activate(tool);
            }
            redraw(target);
        });
        glfwSetCharCallback(window, [](GLFWwindow* target, unsigned int codepoint) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            if (!state.console.opened) return;
            if (codepoint >= 32 && codepoint <= 126) {
                char letter = static_cast<char>(codepoint);
                state.console.insert(std::string_view(&letter, 1));
            } else state.console.feedback = "Use one line of ASCII text";
            redraw(target);
        });
        glfwSetKeyCallback(window, [](GLFWwindow* target, int key, int, int action, int mods) {
            if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
                state.console.opened = !state.console.opened;
                state.hover = Tool::None;
                redraw(target);
                return;
            }
            // while the console is open, keyboard input edits commands instead of controlling the game.
            if (state.console.opened) {
                if (key == GLFW_KEY_ESCAPE) state.console.opened = false;
                else if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) {
                    if (action == GLFW_PRESS) state.runCommand(state.console.submit());
                } else if ((mods & (GLFW_MOD_CONTROL | GLFW_MOD_SUPER)) && key == GLFW_KEY_V) {
                    if (action == GLFW_PRESS) {
                        const char* clipboard = glfwGetClipboardString(target);
                        if (clipboard) state.console.insert(clipboard);
                    }
                } else if ((mods & (GLFW_MOD_CONTROL | GLFW_MOD_SUPER)) && key == GLFW_KEY_U) {
                    state.console.edit(Edit::Clear);
                } else {
                    switch (key) {
                        case GLFW_KEY_LEFT: state.console.edit(Edit::Left); break;
                        case GLFW_KEY_RIGHT: state.console.edit(Edit::Right); break;
                        case GLFW_KEY_HOME: state.console.edit(Edit::Home); break;
                        case GLFW_KEY_END: state.console.edit(Edit::End); break;
                        case GLFW_KEY_BACKSPACE: state.console.edit(Edit::Backspace); break;
                        case GLFW_KEY_DELETE: state.console.edit(Edit::Delete); break;
                    }
                }
                redraw(target);
                return;
            }
            if (action != GLFW_PRESS) return;
            if (key >= GLFW_KEY_1 && key <= GLFW_KEY_3) {
                state.perspective = key - GLFW_KEY_1;
                state.update();
            }
            if (key == GLFW_KEY_SPACE) state.activate(Tool::Playback);
            if (key == GLFW_KEY_PERIOD) state.activate(Tool::Step);
            if (key == GLFW_KEY_R) state.activate(Tool::Reset);
            if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(target, GLFW_TRUE);
            redraw(target);
        });

        redraw(window);
        // advance the game by one half-turn when the timer expires.
        while (!app.closing && !glfwWindowShouldClose(window)) {
            if (app.clock.consume()) {
                app.advance();
                redraw(window);
            }
            if (app.clock.running()) {
                double seconds = app.clock.wait();
                if (seconds > 0) glfwWaitEventsTimeout(seconds);
                else glfwPollEvents();
            } else glfwWaitEvents();
        }
        glfwSetWindowRefreshCallback(window, nullptr);
        glfwSetFramebufferSizeCallback(window, nullptr);
        glfwSetCursorPosCallback(window, nullptr);
        glfwSetCursorEnterCallback(window, nullptr);
        glfwSetMouseButtonCallback(window, nullptr);
        glfwSetCharCallback(window, nullptr);
        glfwSetKeyCallback(window, nullptr);
        glfwSetWindowUserPointer(window, nullptr);
        return 0;
    }
}

int main() {
    glfwSetErrorCallback([](int code, const char* message) {
        std::cerr << "GLFW " << code << ": " << message << '\n';
    });
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(960, 800, "Generals", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "OpenGL loading failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    glfwSwapInterval(1);
    glfwSetWindowSizeLimits(window, 480, 420, GLFW_DONT_CARE, GLFW_DONT_CARE);

    int result = NEBULA::run(window);
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
