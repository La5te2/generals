// application entry point: bring the game components together into an interactive desktop application.
// create the window and OpenGL context, handle page navigation, and route keyboard, mouse and console input.
// coordinate session controls and display updates, then release resources when the application closes.
#include "local.hpp"
#include "dialog.hpp"
#include "renderer.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <commctrl.h>
#include <dwmapi.h>
#endif
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

namespace NEBULA {
    struct WindowState {
        LocalMatch local;
        Renderer renderer;
        Console console;
        Setup setup;
        LocalSnapshot displayed = local.snapshot();
        bool showingBoard = false, closing = false;
        bool repaintTimer = false; // a native timer redraws messages and live games during window drags.
        bool painting = false;
#ifdef _WIN32
        bool desktopPacing = false; // during a window drag, wait for desktop composition instead of a separate swap interval.
#endif
        bool observingLocal = false;
        int perspective = 2; // 0 = red, 1 = blue, 2 = full board.
        Tool hover = Tool::None;

        bool active() const { return displayed.state == LocalState::Active; }
        bool running() const { return displayed.running; }
        const Observation& view() const { return (*displayed.views)[perspective]; }
        void update() { displayed = local.snapshot(); }

        // acquire one published snapshot, keeping the board, playback status and result consistent for the whole frame.
        bool refresh() {
            if (!showingBoard) return false;
            auto next = local.snapshot();
            bool changed = next.views != displayed.views || next.state != displayed.state || next.running != displayed.running;
            displayed = std::move(next);
            if (observingLocal && !active()) {
                observingLocal = false;
                const std::string& error = displayed.error;
                setup.notify(!error.empty() ? error : view().result == Phases::RedWin ? "Red wins"
                           : view().result == Phases::BlueWin ? "Blue wins" : "Draw");
                if (!error.empty()) showingBoard = false;
                hover = Tool::None;
                changed = true;
            }
            return changed;
        }

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
            if (!local.start({setup.fields[3].input, setup.fields[4].input}, seed(), setup.milliseconds)) {
                update();
                setup.notify(displayed.error);
                return;
            }
            setup.message.clear();
            setup.focus = -1;
            setup.fileHover = -1;
            showingBoard = true;
            observingLocal = true;
            perspective = 2;
            hover = Tool::None;
            update();
        }

        // end the current match and return to its configuration page, keeping the selected mode.
        void stop() {
            if (active()) {
                local.stop();
                update();
                setup.notify("Match stopped");
            }
            showingBoard = false;
            observingLocal = false;
            hover = Tool::None;
        }

        // stop the match first, then return to the home page and release text input focus.
        void back() {
            stop();
            setup.scene = Scene::Home;
            setup.focus = -1;
            setup.fileHover = -1;
            setup.message.clear();
            console.opened = false;
            console.edit(Edit::Clear);
        }

        void quit() { stop(); closing = true; }

        bool setTurn(int milliseconds) {
            if (!local.setInterval(milliseconds)) return false;
            setup.milliseconds = milliseconds;
            return true;
        }

        // console input calls the same application operations as window controls and keyboard shortcuts.
        // those operations exist independently of the console parser.
        void runCommand(ParsedCommand command) {
            if (command.type == Command::Back) { back(); return; }
            if (command.type == Command::Quit) { quit(); return; }
            if (command.type == Command::Turn) {
                if (setTurn(command.milliseconds)) console.feedback = "Local half-turn: " + std::to_string(setup.milliseconds) + " ms";
                return;
            }
            if (command.type == Command::Help) {
                console.feedback = "help back quit. TURN milliseconds: positive integer. Current: " + std::to_string(setup.milliseconds) + " ms";
                return;
            }
            if (command.type == Command::Invalid) console.feedback = "Invalid command. Use help, back, quit or TURN followed by positive milliseconds";
        }

        // board buttons and keyboard shortcuts control playback independently of console commands.
        void playback() {
            if (!active()) return;
            if (running()) local.pause();
            else local.resume();
            refresh();
        }

        void step() {
            if (!active() || running()) return;
            local.advance();
            refresh();
        }

        void browse(GLFWwindow* window, int field) {
            std::string error;
            bool program = setup.scene != Scene::Replay;
            setup.fileHover = -1;
            auto selected = chooseFile(window, program, error);
            if (!selected) { if (!error.empty()) setup.notify(error); return; }
            std::error_code status;
            std::filesystem::path path(std::u8string(selected->begin(), selected->end()));
            if (!std::filesystem::is_regular_file(path, status)) {
                setup.notify("Select an existing file"); return;
            }
            // program fields are command lines, while replay fields hold an unquoted file path.
            std::string value = *selected;
            if (program) {
                char quote = value.find('"') == std::string::npos ? '"' : '\'';
                if (value.find(quote) != std::string::npos) {
                    setup.notify("Choose a program path with at most one kind of quote"); return;
                }
                value = quote + value + quote;
            }
            TextInput next;
            if (!next.insert(value)) { setup.notify(next.feedback); return; }
            setup.fields[field] = std::move(next);
            setup.focus = field;
            setup.message.clear();
        }

        // select an action by testing the mouse position against the rectangles used for drawing.
        void click(GLFWwindow* window, double x, double y, int width, int height) {
            if (console.opened) return;
            float scale = barScale(height);
            if (setup.scene != Scene::Home && backButton(scale).contains(x, y)) { back(); return; }
            if (showingBoard) {
                for (int mode = 0; mode < 3; ++mode) {
                    if (viewButton(mode, scale).contains(x, y)) { perspective = mode; update(); }
                }
                if (!active()) {
                    if (setupButton(width, scale).contains(x, y)) showingBoard = false;
                    return;
                }
                if (toolButton(Tool::Playback, width, scale).contains(x, y)) {
                    playback();
                }
                if (toolButton(Tool::Step, width, scale).contains(x, y)) step();
                if (toolButton(Tool::Stop, width, scale).contains(x, y)) stop();
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
            for (int field : inputOrder(setup.scene)) {
                if (hasFileButton(setup.scene, field) && fileButton(setup.scene, field, width, height).contains(x, y)) {
                    browse(window, field);
                    return;
                }
            }
            setup.focus = -1;
            if (setup.scene == Scene::Online) {
                for (int server = 0; server < 2; ++server) {
                    if (choiceButton(formControl(setup.scene, 0, width, height), server, 2).contains(x, y)) {
                        setup.mainServer = server == 1;
                    }
                }
            }
            for (int field : inputOrder(setup.scene)) {
                if (field >= 0 && inputField(setup.scene, field, width, height).contains(x, y)) setup.focus = field;
            }
            if (setup.focus >= 0) setup.fields[setup.focus].edit(Edit::End);
            if (startButton(width, scale).contains(x, y)) start();
        }

        // keyboard text goes to the console while open, otherwise to the focused configuration field.
        TextInput* input() {
            if (console.opened) return &console;
            if (!showingBoard && setup.focus >= 0) return &setup.fields[setup.focus];
            return nullptr;
        }
    };

    void drawWindow(GLFWwindow* window);

    // input can share a pending paint. resizing paints immediately, using the current client dimensions.
    void redraw(GLFWwindow* window, bool immediate = false) {
#ifdef _WIN32
        auto* app = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
        UINT flags = RDW_INVALIDATE | RDW_NOERASE;
        if (immediate && app && !app->painting) flags |= RDW_UPDATENOW;
        RedrawWindow(glfwGetWin32Window(window), nullptr, nullptr, flags);
#else
        (void)immediate;
        drawWindow(window);
#endif
    }

#ifdef _WIN32
    // timer callbacks also run while Windows handles a window drag inside event processing.
    void CALLBACK repaintWindow(HWND, UINT, UINT_PTR id, DWORD) {
        redraw(reinterpret_cast<GLFWwindow*>(id));
    }

    // preserve GLFW's message handling and change presentation timing only while the user moves or resizes the window.
    LRESULT CALLBACK windowMessages(HWND handle, UINT message, WPARAM first, LPARAM second, UINT_PTR, DWORD_PTR data) {
        auto* window = reinterpret_cast<GLFWwindow*>(data);
        auto& app = *static_cast<WindowState*>(glfwGetWindowUserPointer(window));
        if (message == WM_ENTERSIZEMOVE) {
            BOOL composed = FALSE;
            if (SUCCEEDED(DwmIsCompositionEnabled(&composed)) && composed) {
                app.desktopPacing = true;
                glfwSwapInterval(0);
            }
        } else if (message == WM_EXITSIZEMOVE) {
            if (app.desktopPacing) glfwSwapInterval(1);
            app.desktopPacing = false;
        }
        LRESULT result = DefSubclassProc(handle, message, first, second);
        if (message == WM_EXITSIZEMOVE) redraw(window, true);
        return result;
    }
#endif

    // GLFW sends its Windows refresh callback from WM_PAINT. keep the complete frame inside that paint operation.
    void refreshWindow(GLFWwindow* window) {
        auto* app = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
        if (!app || app->painting) return;
        app->painting = true;
#ifdef _WIN32
        HWND handle = glfwGetWin32Window(window);
        // an immediate size repaint may already have consumed the pending paint region.
        if (!GetUpdateRect(handle, nullptr, FALSE)) { app->painting = false; return; }
        PAINTSTRUCT paint{};
        BeginPaint(handle, &paint);
        drawWindow(window);
        EndPaint(handle, &paint);
#else
        drawWindow(window);
#endif
        app->painting = false;
    }

    // draw the current page from stored state. the session timer and board controls advance the game separately.
    void drawWindow(GLFWwindow* window) {
        auto* state = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
        if (!state) return;
        auto& app = *state;
        app.refresh();
        // clear expired text here so both event callbacks and the outer loop erase its final frame.
        if (!app.showingBoard && app.setup.messageOpacity() == 0) app.setup.message.clear();
#ifdef _WIN32
        HWND handle = glfwGetWin32Window(window);
        UINT_PTR id = reinterpret_cast<UINT_PTR>(window);
        bool live = app.showingBoard && app.running();
        if (live || (!app.showingBoard && !app.setup.message.empty())) {
            double delay = live ? 1.0 / 60 : app.setup.messageWait();
            UINT milliseconds = static_cast<UINT>(delay * 1000) + 1;
            app.repaintTimer = SetTimer(handle, id, milliseconds, repaintWindow) != 0;
        } else if (app.repaintTimer) {
            KillTimer(handle, id);
            app.repaintTimer = false;
        }
#endif
        int width, height, pixelsWide, pixelsHigh;
        glfwGetWindowSize(window, &width, &height);
        glfwGetFramebufferSize(window, &pixelsWide, &pixelsHigh);
        if (pixelsWide == 0 || pixelsHigh == 0) return;
        // window coordinates place controls. framebuffer dimensions set the high-DPI viewport.
        glViewport(0, 0, pixelsWide, pixelsHigh);
        if (app.showingBoard) {
            app.renderer.draw(app.view(), app.perspective, width, height, app.running(), app.active(), app.hover, app.console);
        } else app.renderer.drawSetup(app.setup, width, height, app.console);
        // draw() or drawSetup() fills the back buffer, then this swap presents the completed frame.
        glfwSwapBuffers(window);
#ifdef _WIN32
        // pace drag frames with desktop composition. the swap interval is disabled above to avoid two separate waits.
        if (app.desktopPacing && FAILED(DwmFlush())) {
            app.desktopPacing = false;
            glfwSwapInterval(1);
        }
#endif
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
#ifdef _WIN32
        if (!SetWindowSubclass(glfwGetWin32Window(window), windowMessages, 1, reinterpret_cast<DWORD_PTR>(window))) {
            std::cerr << "Window resize handler setup failed\n";
            glfwSetWindowUserPointer(window, nullptr);
            return 1;
        }
#endif
        // moving or resizing a window can hold execution inside GLFW's event handling.
        // refresh callbacks redraw there, while the outer loop waits for event handling to return.
        glfwSetWindowRefreshCallback(window, refreshWindow);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* target, int, int) { redraw(target, true); });
        glfwSetWindowCloseCallback(window, [](GLFWwindow* target) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            state.quit();
        });
        // redraw a tooltip when the hovered tool changes, keeping ordinary mouse motion inexpensive.
        glfwSetCursorPosCallback(window, [](GLFWwindow* target, double x, double y) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            Tool hover = Tool::None;
            int fileHover = -1;
            int width, height;
            glfwGetWindowSize(target, &width, &height);
            if (state.showingBoard && state.active() && !state.console.opened) {
                for (Tool tool : {Tool::Playback, Tool::Step, Tool::Stop}) {
                    if (toolButton(tool, width, barScale(height)).contains(x, y)) hover = tool;
                }
            } else if (!state.showingBoard && !state.console.opened) {
                for (int field : inputOrder(state.setup.scene)) {
                    if (hasFileButton(state.setup.scene, field) && fileButton(state.setup.scene, field, width, height).contains(x, y)) {
                        fileHover = field;
                    }
                }
            }
            if (hover != state.hover || fileHover != state.setup.fileHover) {
                state.hover = hover;
                state.setup.fileHover = fileHover;
                redraw(target);
            }
        });
        glfwSetCursorEnterCallback(window, [](GLFWwindow* target, int entered) {
            if (entered) return;
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            state.hover = Tool::None;
            state.setup.fileHover = -1;
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
            state.click(target, x, y, width, height);
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
                state.setup.fileHover = -1;
            } else if (state.console.opened) {
                if (key == GLFW_KEY_ESCAPE) state.console.opened = false;
                else if ((key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) && action == GLFW_PRESS) {
                    state.runCommand(state.console.submit());
                } else editInput(target, state.console, key, action, mods);
            } else if (!state.showingBoard) {
                if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) state.back();
                else if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
                    auto order = inputOrder(state.setup.scene);
                    auto current = std::find(order.begin(), order.end(), state.setup.focus);
                    do {
                        if (current == order.end() || ++current == order.end()) current = order.begin();
                    } while (*current < 0 && state.setup.scene != Scene::Home);
                    state.setup.focus = *current;
                    if (state.setup.focus >= 0) state.setup.fields[state.setup.focus].edit(Edit::End);
                } else if (auto* input = state.input()) {
                    editInput(target, *input, key, action, mods);
                    state.setup.notify(input->feedback);
                }
            } else if (action == GLFW_PRESS) {
                if (key >= GLFW_KEY_1 && key <= GLFW_KEY_3) { state.perspective = key - GLFW_KEY_1; state.update(); }
                if (key == GLFW_KEY_SPACE) state.playback();
                if (key == GLFW_KEY_PERIOD) state.step();
                if (key == GLFW_KEY_ESCAPE) state.back();
            }
            redraw(target);
        });

        redraw(window);
        // observe the local session without advancing it. its clock runs independently of this window loop.
        while (!app.closing && !glfwWindowShouldClose(window)) {
            bool repaint = app.refresh();
            if (!app.repaintTimer && !app.showingBoard && !app.setup.message.empty()) {
                if (app.setup.messageOpacity() < 1) repaint = true;
            }
            if (repaint) redraw(window);

            // wait for the earlier timer. a negative delay means that only input can wake the window.
            double seconds = app.showingBoard && app.running() && !app.repaintTimer ? 1.0 / 60 : -1;
            if (!app.repaintTimer && !app.showingBoard && !app.setup.message.empty()) {
                double messageDelay = app.setup.messageWait();
                seconds = seconds < 0 ? messageDelay : std::min(seconds, messageDelay);
            }
            if (seconds > 0) glfwWaitEventsTimeout(seconds);
            else if (seconds == 0) glfwPollEvents();
            else glfwWaitEvents();
        }
#ifdef _WIN32
        if (app.repaintTimer) KillTimer(glfwGetWin32Window(window), reinterpret_cast<UINT_PTR>(window));
        RemoveWindowSubclass(glfwGetWin32Window(window), windowMessages, 1);
        if (app.desktopPacing) glfwSwapInterval(1);
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
