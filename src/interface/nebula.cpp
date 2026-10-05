// application entry point: bring match control, input handling and rendering together in one event loop.

#include "clock.hpp"
#include "match.hpp"
#include "renderer.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif
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
        Observation view; // a copy of the selected perspective, refreshed after a move or a view change.
        bool showingBoard = false, closing = false;
        bool messageTimer = false; // a native timer supplies message redraws on Windows.
        int perspective = 2; // 0 = red, 1 = blue, 2 = full board.
        Tool hover = Tool::None;

        bool active() const { return match.state() == MatchState::Active; }
        void update() { view = match.view(perspective); }

        // validate the selected mode before replacing the match and starting its timer.
        void start() {
            if (active()) {
                setup.notify("A match is already active");
                return;
            }
            if (setup.scene == Scene::Online) {
                setup.notify(setup.fields[0].input.empty() || setup.fields[1].input.empty()
                    ? "Username and user ID required" : "Online transport is unavailable");
                return;
            }
            if (setup.scene == Scene::Replay) {
                std::filesystem::path path(setup.fields[2].input);
                std::error_code error;
                if (!std::filesystem::is_regular_file(path, error) || !std::ifstream(path, std::ios::binary)) {
                    setup.notify("Replay file cannot be opened");
                } else setup.notify("Replay playback is unavailable");
                return;
            }
            if (setup.scene != Scene::Local) {
                setup.notify("Select a mode first");
                return;
            }
            std::random_device seed;
            if (!match.start({Participant::Builtin, Participant::Builtin}, seed())) {
                setup.notify("Map generation failed");
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

        // end the current match and return to its configuration page, keeping the selected mode.
        void stop() {
            clock.pause();
            if (active()) {
                match.stop();
                setup.notify("Match stopped");
            }
            showingBoard = false;
            hover = Tool::None;
        }

        // stop the match first, then return to the home page and release text input focus.
        void back() {
            stop();
            setup.scene = Scene::Home;
            setup.focus = -1;
            setup.message.clear();
            console.opened = false;
            console.edit(Edit::Clear);
        }

        // resolve one half-turn, update the displayed observation and pause at the end of the game.
        void advance() {
            if (match.advance()) update();
            if (!active()) {
                clock.pause();
                setup.notify(view.result == Phases::RedWin ? "Red wins"
                           : view.result == Phases::BlueWin ? "Blue wins" : "Draw");
                hover = Tool::None;
            }
        }

        // console.cpp parses text. this function applies the resulting command to the window's state.
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

        // select an action by testing the mouse position against the rectangles used for drawing.
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

        // keyboard text goes to the console while open, otherwise to the focused configuration field.
        TextInput* input() {
            if (console.opened) return &console;
            if (!showingBoard && setup.focus >= 0) return &setup.fields[setup.focus];
            return nullptr;
        }
    };

    void redraw(GLFWwindow* window);

#ifdef _WIN32
    // timer callbacks also run while Windows handles a window drag inside event processing.
    void CALLBACK repaintMessage(HWND, UINT, UINT_PTR id, DWORD) {
        redraw(reinterpret_cast<GLFWwindow*>(id));
    }
#endif

    // draw the current page from stored state. the timer and commands advance the game separately.
    void redraw(GLFWwindow* window) {
        auto* state = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
        if (!state) return;
        auto& app = *state;
        // clear expired text here so both event callbacks and the outer loop erase its final frame.
        if (!app.showingBoard && app.setup.messageOpacity() == 0) app.setup.message.clear();
#ifdef _WIN32
        HWND handle = glfwGetWin32Window(window);
        UINT_PTR id = reinterpret_cast<UINT_PTR>(window);
        if (!app.showingBoard && !app.setup.message.empty()) {
            UINT milliseconds = static_cast<UINT>(app.setup.messageWait() * 1000) + 1;
            app.messageTimer = SetTimer(handle, id, milliseconds, repaintMessage) != 0;
        } else if (app.messageTimer) {
            KillTimer(handle, id);
            app.messageTimer = false;
        }
#endif
        int width, height, pixelsWide, pixelsHigh;
        glfwGetWindowSize(window, &width, &height);
        glfwGetFramebufferSize(window, &pixelsWide, &pixelsHigh);
        if (pixelsWide == 0 || pixelsHigh == 0) return;
        // window coordinates place controls. framebuffer dimensions set the high-DPI viewport.
        glViewport(0, 0, pixelsWide, pixelsHigh);
        if (app.showingBoard) {
            app.renderer.draw(app.view, app.perspective, width, height, app.clock.running(), app.active(), app.hover, app.console);
        } else app.renderer.drawSetup(app.setup, width, height, app.console);
        // draw() or drawSetup() fills the back buffer, then this swap presents the completed frame.
        glfwSwapBuffers(window);
    }

    // editing keys and clipboard paste share this handler for console and configuration inputs.
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
        // store app's address on the GLFW window so each callback can recover the same object.
        glfwSetWindowUserPointer(window, &app);
        // moving or resizing a window can hold execution inside GLFW's event handling.
        // refresh callbacks redraw there, while the outer loop waits for event handling to return.
        glfwSetWindowRefreshCallback(window, redraw);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* target, int, int) { redraw(target); });
        glfwSetWindowCloseCallback(window, [](GLFWwindow* target) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            state.stop();
        });
        // redraw a tooltip when the hovered tool changes, keeping ordinary mouse motion inexpensive.
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
        // mouse events supply window coordinates, matching the layout functions in scene.hpp.
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
        // GLFW's character callback supplies typed text after applying the keyboard layout.
        // editing keys and shortcuts are handled by the key callback below.
        glfwSetCharCallback(window, [](GLFWwindow* target, unsigned int codepoint) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            TextInput* input = state.input();
            if (!input) return;
            if (codepoint >= 32 && codepoint <= 126) {
                char letter = static_cast<char>(codepoint);
                input->insert(std::string_view(&letter, 1));
            } else input->feedback = "Use one line of ASCII text";
            if (!state.console.opened) state.setup.notify(input->feedback);
            redraw(target);
        });
        // route keys in this order: console toggle, console input, setup input, board controls.
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
                    state.setup.notify(input->feedback);
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
        // game steps and fading messages request redraws independently. input callbacks also redraw.
        while (!app.closing && !glfwWindowShouldClose(window)) {
            bool repaint = app.clock.consume();
            if (repaint) app.advance();
            if (!app.messageTimer && !app.showingBoard && !app.setup.message.empty()) {
                if (app.setup.messageOpacity() < 1) repaint = true;
            }
            if (repaint) redraw(window);

            // wait for the earlier timer. a negative delay means that only input can wake the window.
            double seconds = app.clock.running() ? app.clock.wait() : -1;
            if (!app.messageTimer && !app.showingBoard && !app.setup.message.empty()) {
                double messageDelay = app.setup.messageWait();
                seconds = seconds < 0 ? messageDelay : std::min(seconds, messageDelay);
            }
            if (seconds > 0) glfwWaitEventsTimeout(seconds);
            else if (seconds == 0) glfwPollEvents();
            else glfwWaitEvents();
        }
#ifdef _WIN32
        if (app.messageTimer) KillTimer(glfwGetWin32Window(window), reinterpret_cast<UINT_PTR>(window));
#endif
        // detach callbacks before app is destroyed, ending their access to its stack address.
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
    // request an OpenGL 3.3 core context and multisampling to smooth triangle edges.
    // GLFW creates a double-buffered window by default.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(960, 800, "Generals", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    // make this context current before GLAD loads the OpenGL function addresses for it.
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "OpenGL loading failed\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    // request swaps synchronized with the display refresh. actual redraws are scheduled in run().
    glfwSwapInterval(1);
    glfwSetWindowSizeLimits(window, 540, 560, GLFW_DONT_CARE, GLFW_DONT_CARE);
    int result = NEBULA::run(window);
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
