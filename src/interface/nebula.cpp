#include "clock.hpp"
#include "match.hpp"
#include "renderer.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

namespace NEBULA {
    struct WindowState {
        Match match;
        Renderer renderer;
        Clock clock;
        Console console;
        Setup setup;
        Observation view;
        bool showingBoard = false, closing = false;
        int perspective = 2;
        Tool hover = Tool::None;

        bool active() const { return match.state() == MatchState::Active; }
        void update() { view = match.view(perspective); }

        void start() {
            if (active()) {
                setup.message = "A match is already active";
                return;
            }
            if (setup.scene == Scene::Online) {
                setup.message = setup.fields[0].input.empty() || setup.fields[1].input.empty()
                    ? "Username and user ID required" : "Online transport is unavailable";
                return;
            }
            if (setup.scene == Scene::Replay) {
                std::filesystem::path path(setup.fields[2].input);
                std::error_code error;
                if (!std::filesystem::is_regular_file(path, error) || !std::ifstream(path, std::ios::binary)) {
                    setup.message = "Replay file cannot be opened";
                } else setup.message = "Replay playback is unavailable";
                return;
            }
            if (setup.scene != Scene::Local) {
                setup.message = "Select a mode first";
                return;
            }
            std::random_device seed;
            if (!match.start({Participant::Builtin, Participant::Builtin}, seed())) {
                setup.message = "Map generation failed";
                return;
            }
            setup.message.clear();
            setup.focus = -1;
            showingBoard = true;
            perspective = 2;
            hover = Tool::None;
            clock.reset();
            update();
            clock.resume();
        }

        void stop() {
            clock.pause();
            if (active()) {
                match.stop();
                setup.message = "Match stopped";
            }
            showingBoard = false;
            hover = Tool::None;
        }

        void back() {
            stop();
            setup.scene = Scene::Home;
            setup.focus = -1;
            setup.message.clear();
            console.opened = false;
            console.edit(Edit::Clear);
        }

        void advance() {
            if (match.advance()) update();
            if (!active()) {
                clock.pause();
                setup.message = view.result == Phases::RedWin ? "Red wins"
                              : view.result == Phases::BlueWin ? "Blue wins" : "Draw";
                hover = Tool::None;
            }
        }

        void runCommand(Command command) {
            if (command == Command::Start) { start(); console.feedback = setup.message; return; }
            if (command == Command::Stop) { stop(); console.feedback = "Stopped"; return; }
            if (command == Command::Back) { back(); return; }
            if (command == Command::Quit) { stop(); closing = true; return; }
            if (command == Command::Help) {
                console.feedback = "help start stop back pause resume step quit";
                return;
            }
            if (command == Command::None) return;
            if (command == Command::Invalid) { console.feedback = "Unknown command or extra arguments"; return; }
            if (!active()) { console.feedback = "Start a local match first"; return; }
            if (command == Command::Pause) { clock.pause(); console.feedback = "Paused"; }
            if (command == Command::Resume) { clock.resume(); console.feedback = "Running"; }
            if (command == Command::Step) {
                if (clock.running()) console.feedback = "Pause the game before stepping";
                else {
                    clock.reset();
                    advance();
                    console.feedback = active() ? "Advanced one half-turn" : setup.message;
                }
            }
        }

        void click(double x, double y, int width, int height) {
            if (console.opened) return;
            if (setup.scene != Scene::Home && backButton().contains(x, y)) { back(); return; }
            if (showingBoard) {
                for (int mode = 0; mode < 3; ++mode) {
                    if (viewButton(mode).contains(x, y)) { perspective = mode; update(); }
                }
                if (!active()) {
                    if (setupButton(width).contains(x, y)) showingBoard = false;
                    return;
                }
                if (toolButton(Tool::Playback, width).contains(x, y)) {
                    runCommand(clock.running() ? Command::Pause : Command::Resume);
                }
                if (toolButton(Tool::Step, width).contains(x, y)) runCommand(Command::Step);
                if (toolButton(Tool::Stop, width).contains(x, y)) stop();
                return;
            }
            if (setup.scene == Scene::Home) {
                for (int index = 0; index < 3; ++index) {
                    if (menuButton(index, width, height).contains(x, y)) {
                        setup.scene = static_cast<Scene>(index + 1);
                        setup.message.clear();
                    }
                }
                return;
            }
            setup.focus = -1;
            if (setup.scene == Scene::Online) {
                for (int server = 0; server < 2; ++server) {
                    if (choiceButton(formControl(setup.scene, 0, width, height), server, 2).contains(x, y)) {
                        setup.mainServer = server == 1;
                    }
                }
                for (int field = 0; field < 2; ++field) {
                    if (inputField(setup.scene, field, width, height).contains(x, y)) setup.focus = field;
                }
            }
            if (setup.scene == Scene::Replay && inputField(setup.scene, 2, width, height).contains(x, y)) setup.focus = 2;
            if (setup.focus >= 0) setup.fields[setup.focus].edit(Edit::End);
            if (startButton(width).contains(x, y)) start();
        }

        TextInput* input() {
            if (console.opened) return &console;
            if (!showingBoard && setup.focus >= 0) return &setup.fields[setup.focus];
            return nullptr;
        }
    };

    void redraw(GLFWwindow* window) {
        auto& app = *static_cast<WindowState*>(glfwGetWindowUserPointer(window));
        int width, height, pixelsWide, pixelsHigh;
        glfwGetWindowSize(window, &width, &height);
        glfwGetFramebufferSize(window, &pixelsWide, &pixelsHigh);
        if (pixelsWide == 0 || pixelsHigh == 0) return;
        // window coordinates place controls. framebuffer dimensions set the high-DPI viewport.
        glViewport(0, 0, pixelsWide, pixelsHigh);
        if (app.showingBoard) {
            app.renderer.draw(app.view, app.perspective, width, height, app.clock.running(), app.active(), app.hover, app.console);
        } else app.renderer.drawSetup(app.setup, width, height, app.console);
        glfwSwapBuffers(window);
    }

    void editInput(GLFWwindow* window, TextInput& input, int key, int action, int mods) {
        if ((mods & (GLFW_MOD_CONTROL | GLFW_MOD_SUPER)) && key == GLFW_KEY_V) {
            if (action == GLFW_PRESS) {
                const char* clipboard = glfwGetClipboardString(window);
                if (clipboard) input.insert(clipboard);
            }
        } else if ((mods & (GLFW_MOD_CONTROL | GLFW_MOD_SUPER)) && key == GLFW_KEY_U) input.edit(Edit::Clear);
        else {
            switch (key) {
                case GLFW_KEY_LEFT: input.edit(Edit::Left); break;
                case GLFW_KEY_RIGHT: input.edit(Edit::Right); break;
                case GLFW_KEY_HOME: input.edit(Edit::Home); break;
                case GLFW_KEY_END: input.edit(Edit::End); break;
                case GLFW_KEY_BACKSPACE: input.edit(Edit::Backspace); break;
                case GLFW_KEY_DELETE: input.edit(Edit::Delete); break;
            }
        }
    }

    int run(GLFWwindow* window) {
        // run() releases the renderer's GPU resources before main() destroys the window.
        WindowState app;
        if (!app.renderer.init()) return 1;
        glfwSetWindowUserPointer(window, &app);
        glfwSetWindowRefreshCallback(window, redraw);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* target, int, int) { redraw(target); });
        glfwSetWindowCloseCallback(window, [](GLFWwindow* target) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            state.stop();
        });
        glfwSetCursorPosCallback(window, [](GLFWwindow* target, double x, double y) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            Tool hover = Tool::None;
            if (state.showingBoard && state.active() && !state.console.opened) {
                int width, height;
                glfwGetWindowSize(target, &width, &height);
                for (Tool tool : {Tool::Playback, Tool::Step, Tool::Stop}) {
                    if (toolButton(tool, width).contains(x, y)) hover = tool;
                }
            }
            if (hover != state.hover) { state.hover = hover; redraw(target); }
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
            double x, y;
            int width, height;
            glfwGetCursorPos(target, &x, &y);
            glfwGetWindowSize(target, &width, &height);
            state.click(x, y, width, height);
            redraw(target);
        });
        glfwSetCharCallback(window, [](GLFWwindow* target, unsigned int codepoint) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            TextInput* input = state.input();
            if (!input) return;
            if (codepoint >= 32 && codepoint <= 126) {
                char letter = static_cast<char>(codepoint);
                input->insert(std::string_view(&letter, 1));
            } else input->feedback = "Use one line of ASCII text";
            if (!state.console.opened) state.setup.message = input->feedback;
            redraw(target);
        });
        glfwSetKeyCallback(window, [](GLFWwindow* target, int key, int, int action, int mods) {
            if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            if (key == GLFW_KEY_F1 && action == GLFW_PRESS) {
                state.console.opened = !state.console.opened;
                state.hover = Tool::None;
            } else if (state.console.opened) {
                if (key == GLFW_KEY_ESCAPE) state.console.opened = false;
                else if ((key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) && action == GLFW_PRESS) {
                    state.runCommand(state.console.submit());
                } else editInput(target, state.console, key, action, mods);
            } else if (!state.showingBoard) {
                if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) state.back();
                else if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
                    if (state.setup.scene == Scene::Online) state.setup.focus = state.setup.focus == 0 ? 1 : 0;
                    if (state.setup.scene == Scene::Replay) state.setup.focus = 2;
                } else if (auto* input = state.input()) {
                    editInput(target, *input, key, action, mods);
                    state.setup.message = input->feedback;
                }
            } else if (action == GLFW_PRESS) {
                if (key >= GLFW_KEY_1 && key <= GLFW_KEY_3) { state.perspective = key - GLFW_KEY_1; state.update(); }
                if (key == GLFW_KEY_SPACE) state.runCommand(state.clock.running() ? Command::Pause : Command::Resume);
                if (key == GLFW_KEY_PERIOD) state.runCommand(Command::Step);
                if (key == GLFW_KEY_ESCAPE) state.back();
            }
            redraw(target);
        });

        redraw(window);
        while (!app.closing && !glfwWindowShouldClose(window)) {
            if (app.clock.consume()) { app.advance(); redraw(window); }
            if (app.clock.running()) {
                double seconds = app.clock.wait();
                if (seconds > 0) glfwWaitEventsTimeout(seconds);
                else glfwPollEvents();
            } else glfwWaitEvents();
        }
        glfwSetWindowRefreshCallback(window, nullptr);
        glfwSetFramebufferSizeCallback(window, nullptr);
        glfwSetWindowCloseCallback(window, nullptr);
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
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "OpenGL loading failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    glfwSwapInterval(1);
    glfwSetWindowSizeLimits(window, 540, 560, GLFW_DONT_CARE, GLFW_DONT_CARE);
    int result = NEBULA::run(window);
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
