// application entry point: bring the game components together into an interactive desktop application.
// create the window and OpenGL context, handle page navigation, and route keyboard, mouse and console input.
// coordinate session controls and display updates, then release resources when the application closes.
#include "local.hpp"
#include "online.hpp"
#include "dialog.hpp"
#include "renderer.hpp"
#include "manual.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>

namespace NEBULA {
    struct WindowState {
        LocalMatch local;
        OnlineMatch online;
        Replay replay;
        Clock replayClock;
        bool replayActive = false;
        Controller controller;
        Renderer renderer;
        Console console;
        Setup setup;
        MatchSnapshot displayed = local.snapshot();
        bool showingBoard = false, closing = false;
        unsigned int repaintTimer = 0; // native repaint interval in milliseconds. zero means the timer is stopped.
        bool painting = false;
        bool observingMatch = false;
        bool automatic = false;
        std::optional<Clock::Time> resetAt;
        int perspective = 2; // 0 = red, 1 = blue, 2 = full board.
        Tool hover = Tool::None;

        bool active() const { return displayed.state == MatchState::Active; }
        bool running() const { return displayed.running; }
        bool human() const { return active() && controller.player() >= 0; }
        PlayerInput& inputSession() { return setup.scene == Scene::Online ? static_cast<PlayerInput&>(online) : local; }
        BoardControls controls() const {
            return {setup.scene, active(), running(), human(), hover, setup.scene == Scene::Replay && replay.cursor() > 0};
        }
        const Observation& view() const { return (*displayed.views)[perspective]; }
        void update() {
            if (setup.scene == Scene::Replay && replay.loaded()) {
                displayed = {};
                displayed.views = std::make_shared<const std::array<Observation, 3>>(matchViews(replay.state()));
                displayed.names = replay.names();
                displayed.state = replayActive ? MatchState::Active : MatchState::Finished;
                displayed.running = replayClock.running();
            } else displayed = setup.scene == Scene::Online ? online.snapshot() : static_cast<MatchSnapshot>(local.snapshot());
            if (setup.scene == Scene::Online && displayed.player >= 0) {
                perspective = displayed.player;
                if (setup.humanPlayer(5) && controller.player() < 0 && view().cols > 0 && active()) {
                    controller.reset(perspective, view());
                }
            }
            controller.sync(displayed);
        }

        void report(std::string message) {
            if (!displayed.error.empty()) message = displayed.error;
            else if (setup.scene == Scene::Local) {
                auto saved = local.snapshot().saved;
                if (!saved.empty()) message += ". Saved " + saved.filename().string();
            }
            setup.notify(message);
            console.feedback = message;
            if (!displayed.error.empty()) std::cerr << message << '\n';
        }

        // replay time changes the playback cursor. the reader reconstructs the requested position.
        void seekReplay(std::size_t halfTurn) {
            if (!replay.seek(halfTurn)) return;
            if (halfTurn == replay.length()) {
                replayClock.pause();
                replayActive = false;
                update();
                report("Replay ended");
                scheduleReset();
            } else update();
        }

        // acquire one published snapshot, keeping the board, playback status and result consistent for the whole frame.
        bool refresh() {
            if (!showingBoard) return false;
            if (setup.scene == Scene::Replay) {
                if (!replayActive || !replayClock.consume()) return false;
                seekReplay(replay.cursor() + 1);
                return true;
            }
            auto previous = displayed;
            update();
            bool changed = previous.views != displayed.views || previous.state != displayed.state || previous.running != displayed.running ||
                previous.status != displayed.status || previous.error != displayed.error;
            if (observingMatch && !active()) {
                observingMatch = false;
                const std::string& error = displayed.error;
                report(setup.scene == Scene::Online ? displayed.status :
                    view().result == Phases::RedWin ? "Red wins" : view().result == Phases::BlueWin ? "Blue wins" : "Draw");
                if (error.empty()) scheduleReset();
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
            // consume a pending reset before starting, so a failed attempt stays available for manual retry.
            resetAt.reset();
            if (setup.scene == Scene::Online) {
                OnlineConfig config{setup.mainServer ? Server::Main : Server::Bot,
                    setup.fields[0].input, setup.fields[1].input,
                    setup.humanPlayer(5) ? "" : setup.fields[5].input, setup.fields[7].input, setup.fields[8].input};
                if (!online.start(config)) { setup.notify(online.snapshot().error); return; }
                controller.reset(-1, view());
                perspective = 0;
                showingBoard = observingMatch = true;
                setup.message.clear();
                setup.focus = setup.fileHover = -1;
                hover = Tool::None;
                update();
                return;
            }
            if (setup.scene == Scene::Replay) {
                const auto& name = setup.fields[2].input;
                std::string error;
                if (!replay.load(std::filesystem::path(std::u8string(name.begin(), name.end())), error)) {
                    setup.notify(error);
                    return;
                }
                replayClock.interval = std::chrono::milliseconds(setup.milliseconds);
                replayClock.reset();
                replayActive = replay.length() > 0;
                if (replayActive) replayClock.resume();
                setup.message.clear();
                setup.focus = setup.fileHover = -1;
                showingBoard = true;
                observingMatch = false;
                perspective = 2;
                hover = Tool::None;
                controller.reset(-1, view());
                update();
                if (!replayActive) { report("Replay ended"); scheduleReset(); }
                return;
            }
            if (setup.scene != Scene::Local) {
                setup.notify("Select a mode first");
                return;
            }
            std::array<std::string, 2> commands{setup.fields[3].input, setup.fields[4].input};
            for (int player = 0; player < 2; ++player) {
                if (setup.humanPlayer(player + 3)) commands[player].clear();
            }
            // one keyboard controls one player. local sessions themselves accept any externally supplied actions.
            if (commands[0].empty() && commands[1].empty()) {
                setup.notify("Choose a program for at least one player");
                return;
            }
            int player = commands[0].empty() ? 0 : commands[1].empty() ? 1 : -1;
            const auto& name = setup.fields[6].input;
            std::filesystem::path directory;
            if (name.find_first_not_of(" \t\r\n") != std::string::npos) directory = std::u8string(name.begin(), name.end());
            std::random_device seed;
            if (!local.start(commands, seed(), setup.milliseconds, directory)) {
                update();
                setup.notify(displayed.error);
                return;
            }
            setup.message.clear();
            setup.focus = -1;
            setup.fileHover = -1;
            showingBoard = true;
            observingMatch = true;
            perspective = player >= 0 ? player : 2;
            hover = Tool::None;
            update();
            controller.reset(player, view());
        }

        // stop keeps the final position on screen. reset starts another session with the current configuration.
        void stop() {
            if (!showingBoard) return;
            if (active()) {
                if (setup.scene == Scene::Replay) { replayClock.pause(); replayActive = false; }
                else if (setup.scene == Scene::Online) online.stop();
                else local.stop();
                update();
                report(setup.scene == Scene::Replay ? "Replay stopped" : "Match stopped");
                if (displayed.error.empty()) scheduleReset();
            }
            observingMatch = false;
            controller.deselect();
            hover = Tool::None;
        }

        void reset() {
            if (showingBoard && controls().enabled(Tool::Reset)) start();
        }

        void scheduleReset() {
            resetAt.reset();
            if (automatic && showingBoard && !active() && !closing && displayed.error.empty()) {
                resetAt = Clock::Source::now() + std::chrono::milliseconds(1000);
            }
        }

        void setAuto(bool enabled) {
            automatic = enabled;
            scheduleReset();
        }

        // return one page at a time: board to configuration, then configuration to home.
        void back() {
            if (showingBoard) {
                stop();
                showingBoard = false;
                controller.reset(-1, view());
            } else { setup.scene = Scene::Home; setup.message.clear(); }
            resetAt.reset();
            setup.focus = -1;
            setup.fileHover = -1;
            console.opened = false;
            console.edit(Edit::Clear);
        }

        void quit() {
            stop();
            resetAt.reset();
            // keep a failed recording available for directory correction before releasing the session.
            if (local.snapshot().unsaved) {
                local.stop();
                auto pending = local.snapshot();
                if (pending.unsaved) {
                    setup.notify(pending.error);
                    console.feedback = pending.error;
                    return;
                }
            }
            closing = true;
        }

        bool setTurn(int milliseconds) {
            if (!local.setInterval(milliseconds)) return false;
            setup.milliseconds = milliseconds;
            bool playing = replayClock.running();
            replayClock.interval = std::chrono::milliseconds(milliseconds);
            replayClock.reset();
            if (playing) replayClock.resume();
            return true;
        }

        // size the window relative to its current monitor. the highest level covers that monitor without borders.
        bool setWindow(GLFWwindow* window, int level) {
            if (level < 0 || level > maxWindowLevel) return false;
            int x, y, width, height, count;
            glfwGetWindowPos(window, &x, &y);
            glfwGetWindowSize(window, &width, &height);
            GLFWmonitor** monitors = glfwGetMonitors(&count);
            GLFWmonitor* chosen = glfwGetPrimaryMonitor();
            std::int64_t largest = 0;
            for (int index = 0; index < count; ++index) {
                int left, top, wide, high;
                glfwGetMonitorWorkarea(monitors[index], &left, &top, &wide, &high);
                auto overlap = static_cast<std::int64_t>(std::max(0, std::min(x + width, left + wide) - std::max(x, left))) *
                    std::max(0, std::min(y + height, top + high) - std::max(y, top));
                if (overlap > largest) { largest = overlap; chosen = monitors[index]; }
            }
            if (!chosen) return false;
            if (level == maxWindowLevel) {
                const GLFWvidmode* mode = glfwGetVideoMode(chosen);
                if (!mode) return false;
                // a borderless window fills the monitor while preserving its desktop resolution and refresh rate.
                glfwGetMonitorPos(chosen, &x, &y);
                glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);
                glfwSetWindowSize(window, mode->width, mode->height);
                glfwSetWindowPos(window, x, y);
            } else {
                int left, top, wide, high, frameLeft, frameTop, frameRight, frameBottom;
                glfwGetMonitorWorkarea(chosen, &left, &top, &wide, &high);
                glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_TRUE);
                glfwGetWindowFrameSize(window, &frameLeft, &frameTop, &frameRight, &frameBottom);
                int availableWidth = wide - frameLeft - frameRight;
                int availableHeight = high - frameTop - frameBottom;
                if (availableWidth <= 0 || availableHeight <= 0) return false;
                // levels 0 through 9 use 50% through 95% of the available width and height, centered in the work area.
                double scale = .5 + .5 * level / maxWindowLevel;
                // preserve room for the configuration form, capped by the space available on smaller monitors.
                int nextWidth = std::min(availableWidth, std::max(720, static_cast<int>(availableWidth * scale)));
                int nextHeight = std::min(availableHeight, std::max(560, static_cast<int>(availableHeight * scale)));
                glfwSetWindowSize(window, nextWidth, nextHeight);
                glfwSetWindowPos(window, left + frameLeft + (availableWidth - nextWidth) / 2,
                                 top + frameTop + (availableHeight - nextHeight) / 2);
            }
            hover = Tool::None;
            setup.fileHover = -1;
            return true;
        }

        // console input calls the same application operations as window controls and keyboard shortcuts.
        // those operations exist independently of the console parser.
        void runCommand(GLFWwindow* window, ParsedCommand command) {
            if (command.type == Command::Back) { back(); return; }
            if (command.type == Command::Quit) { quit(); return; }
            if (command.type == Command::Turn) {
                if (setTurn(command.value)) console.feedback = "Local / replay half-turn: " + std::to_string(setup.milliseconds) + " ms";
                return;
            }
            if (command.type == Command::Auto) {
                setAuto(command.value == 1);
                console.feedback = automatic ? "Auto reset: 1000 ms" : "Reset: manual";
                return;
            }
            if (command.type == Command::Win) {
                if (setWindow(window, command.value)) {
                    int width, height;
                    glfwGetWindowSize(window, &width, &height);
                    std::string label = command.value == maxWindowLevel ? "Full screen" : "Window " + std::to_string(command.value);
                    console.feedback = label + ": " +
                        std::to_string(width) + " x " + std::to_string(height);
                } else console.feedback = "Window size update failed";
                return;
            }
            if (command.type == Command::Help) {
                console.feedback = Manual::commands;
                return;
            }
            if (command.type == Command::Man) {
                console.manual = true;
                return;
            }
            if (command.type == Command::Invalid) console.feedback = "Use help for commands or man for the game manual";
        }

        // board buttons and keyboard shortcuts control playback independently of console commands.
        void playback() {
            if (!showingBoard || !controls().enabled(Tool::Playback)) return;
            if (setup.scene == Scene::Replay) {
                if (running()) replayClock.pause();
                else replayClock.resume();
                update();
                return;
            }
            if (running()) local.pause();
            else local.resume();
            refresh();
        }

        void step(Tool direction) {
            if (!showingBoard || !controls().enabled(direction)) return;
            if (setup.scene == Scene::Replay) {
                seekReplay(direction == Tool::Forward ? replay.cursor() + 1 : replay.cursor() - 1);
                replayClock.reset();
                return;
            }
            if (setup.scene != Scene::Local || direction != Tool::Forward) return;
            local.advance();
            refresh();
        }

        void useTool(Tool tool) {
            if (!showingBoard || !controls().enabled(tool)) return;
            switch (tool) {
                case Tool::Backward: case Tool::Forward: step(tool); break;
                case Tool::Playback: playback(); break;
                case Tool::Stop: stop(); break;
                case Tool::Reset: reset(); break;
                default: break;
            }
        }

        void browse(GLFWwindow* window, int field) {
            std::string error;
            PathKind kind = field == 6 ? PathKind::Directory : setup.scene == Scene::Replay ? PathKind::Replay : PathKind::Program;
            bool program = kind == PathKind::Program;
            setup.fileHover = -1;
            auto selected = choosePath(window, kind, error);
            if (!selected) { if (!error.empty()) setup.notify(error); return; }
            std::error_code status;
            std::filesystem::path path(std::u8string(selected->begin(), selected->end()));
            bool valid = kind == PathKind::Directory ? std::filesystem::is_directory(path, status) : std::filesystem::is_regular_file(path, status);
            if (!valid) {
                setup.notify(kind == PathKind::Directory ? "Select an existing directory" : "Select an existing file"); return;
            }
            // only program command lines need quotes. replay and directory fields hold the path itself.
            std::string value = *selected;
            if (program) {
                char quote = value.find('"') == std::string::npos ? '"' : '\'';
                if (value.find(quote) != std::string::npos) {
                    setup.notify("Choose a program path with at most one kind of quote"); return;
                }
                value = quote + value + quote;
            }
            TextInput next;
            // native dialogs return UTF-8 paths. retain those bytes even when the pixel font lacks their glyphs.
            if (value.find_first_of("\r\n") != std::string::npos) { setup.notify("Select a single-line path"); return; }
            next.input = std::move(value);
            next.cursor = next.input.size();
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
                refresh();
                for (Tool tool : tools) {
                    if (toolButton(tool, width, scale).contains(x, y)) { useTool(tool); return; }
                }
                if (human()) {
                    controller.click(inputSession(), displayed, x, y, width, height);
                    update();
                    return;
                }
                for (int mode = 0; controls().spectator() && mode < 3; ++mode) {
                    if (viewButton(mode, scale).contains(x, y)) { perspective = mode; update(); }
                }
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
                        setup.selectServer(server == 1);
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

    // on Windows, input and size changes share a pending paint. their callbacks return before drawing begins.
    void redraw(GLFWwindow* window) {
#ifdef _WIN32
        RedrawWindow(glfwGetWin32Window(window), nullptr, nullptr, RDW_INVALIDATE | RDW_NOERASE);
#else
        drawWindow(window);
#endif
    }

#ifdef _WIN32
    // timer callbacks also run while Windows handles a window drag inside event processing.
    void CALLBACK repaintWindow(HWND, UINT, UINT_PTR id, DWORD) {
        redraw(reinterpret_cast<GLFWwindow*>(id));
    }
#endif

    // GLFW sends its Windows refresh callback from WM_PAINT. keep the complete frame inside that paint operation.
    void refreshWindow(GLFWwindow* window) {
        auto* app = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
        if (!app || app->painting) return;
        app->painting = true;
#ifdef _WIN32
        HWND handle = glfwGetWin32Window(window);
        // a duplicate paint message can arrive after the invalidated region has already been drawn.
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
        if (app.setup.messageOpacity() == 0) app.setup.message.clear();
#ifdef _WIN32
        HWND handle = glfwGetWin32Window(window);
        UINT_PTR id = reinterpret_cast<UINT_PTR>(window);
        bool live = app.showingBoard && (app.running() || (app.setup.scene == Scene::Online && app.active()));
        if (live || !app.setup.message.empty()) {
            double delay = live ? 1.0 / 60 : app.setup.messageWait();
            UINT milliseconds = static_cast<UINT>(delay * 1000) + 1;
            // an unchanged interval keeps its original deadline when other window events request a repaint.
            if (milliseconds != app.repaintTimer) {
                if (SetTimer(handle, id, milliseconds, repaintWindow)) app.repaintTimer = milliseconds;
                else { KillTimer(handle, id); app.repaintTimer = 0; }
            }
        } else if (app.repaintTimer) {
            KillTimer(handle, id);
            app.repaintTimer = 0;
        }
#endif
        int width, height, pixelsWide, pixelsHigh;
        glfwGetWindowSize(window, &width, &height);
#ifdef _WIN32
        // GLFW's Windows client coordinates are framebuffer pixels. one size read keeps layout and viewport together.
        pixelsWide = width;
        pixelsHigh = height;
#else
        glfwGetFramebufferSize(window, &pixelsWide, &pixelsHigh);
#endif
        if (pixelsWide == 0 || pixelsHigh == 0) return;
        // window coordinates place controls. framebuffer dimensions set the high-DPI viewport.
        glViewport(0, 0, pixelsWide, pixelsHigh);
        if (app.showingBoard) {
            std::span<const Action> queued;
            if (app.controller.player() >= 0) queued = app.displayed.queued[app.controller.player()];
            app.renderer.draw(app.view(), app.perspective, width, height, app.controls(), app.setup,
                              app.console, app.controller, queued, app.displayed.names, app.displayed.status);
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
        // run() releases the renderer's GPU resources before the entry function destroys the window.
        WindowState app;
        if (!app.renderer.init()) return 1;
        app.setWindow(window, 4);
        // store app's address on the GLFW window so each callback can recover the same object.
        glfwSetWindowUserPointer(window, &app);
        // moving a window can hold execution inside GLFW's event handling.
        // refresh callbacks redraw there, while the outer loop waits for event handling to return.
        glfwSetWindowRefreshCallback(window, refreshWindow);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* target, int, int) { redraw(target); });
        glfwSetWindowCloseCallback(window, [](GLFWwindow* target) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            state.quit();
            if (!state.closing) glfwSetWindowShouldClose(target, GLFW_FALSE);
        });
        // redraw a tooltip when the hovered tool changes, keeping ordinary mouse motion inexpensive.
        glfwSetCursorPosCallback(window, [](GLFWwindow* target, double x, double y) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            Tool hover = Tool::None;
            int fileHover = -1;
            int width, height;
            glfwGetWindowSize(target, &width, &height);
            if (state.showingBoard && !state.console.opened) {
                for (Tool tool : tools) {
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
            if (action != GLFW_PRESS) return;
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            if (button == GLFW_MOUSE_BUTTON_RIGHT && state.human() && !state.console.opened) {
                state.controller.deselect();
                redraw(target);
                return;
            }
            if (button != GLFW_MOUSE_BUTTON_LEFT) return;
            double x, y;
            int width, height;
            glfwGetCursorPos(target, &x, &y);
            glfwGetWindowSize(target, &width, &height);
            state.click(target, x, y, width, height);
            redraw(target);
        });
        glfwSetScrollCallback(window, [](GLFWwindow* target, double, double vertical) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            if (!state.console.opened || vertical == 0) return;
            int width, height;
            glfwGetWindowSize(target, &width, &height);
            auto layout = consoleLayout(width, height);
            if (state.console.scroll(vertical > 0 ? -3 : 3, layout.columns, layout.rows)) redraw(target);
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
                    state.runCommand(target, state.console.submit());
                } else if (key == GLFW_KEY_UP || key == GLFW_KEY_DOWN || key == GLFW_KEY_PAGE_UP || key == GLFW_KEY_PAGE_DOWN) {
                    int width, height;
                    glfwGetWindowSize(target, &width, &height);
                    auto layout = consoleLayout(width, height);
                    int amount = key == GLFW_KEY_UP || key == GLFW_KEY_DOWN ? 1 : static_cast<int>(std::max(std::size_t{1}, layout.rows - 1));
                    if (key == GLFW_KEY_UP || key == GLFW_KEY_PAGE_UP) amount = -amount;
                    state.console.scroll(amount, layout.columns, layout.rows);
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
            } else if (state.human()) {
                state.refresh();
                if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) state.back();
                else if (key == GLFW_KEY_SPACE) { if (action == GLFW_PRESS) state.playback(); }
                else { state.controller.key(state.inputSession(), state.displayed, key, action, mods); state.update(); }
            } else if (action == GLFW_PRESS) {
                if (state.controls().spectator() && key >= GLFW_KEY_1 && key <= GLFW_KEY_3) {
                    state.perspective = key - GLFW_KEY_1;
                    state.update();
                }
                if (key == GLFW_KEY_SPACE) state.playback();
                if (key == GLFW_KEY_PERIOD) state.useTool(Tool::Forward);
                if (key == GLFW_KEY_COMMA) state.useTool(Tool::Backward);
                if (state.setup.scene == Scene::Replay) {
                    if (key == GLFW_KEY_RIGHT) state.useTool(Tool::Forward);
                    if (key == GLFW_KEY_LEFT) state.useTool(Tool::Backward);
                }
                if (key == GLFW_KEY_ESCAPE) state.back();
            }
            redraw(target);
        });

        // prepare the first frame at the chosen size before making the window visible.
        drawWindow(window);
        glfwShowWindow(window);
        // observe session updates while each session handles its own timing and communication.
        while (!app.closing && !glfwWindowShouldClose(window)) {
            bool repaint = app.refresh();
            if (app.resetAt && Clock::Source::now() >= *app.resetAt) {
                app.reset();
                repaint = true;
            }
            if (!app.repaintTimer && !app.setup.message.empty()) {
                if (app.setup.messageOpacity() < 1) repaint = true;
            }
            if (repaint) redraw(window);

            // wait for the earlier timer. a negative delay means that only input can wake the window.
            bool live = app.showingBoard && (app.running() || (app.setup.scene == Scene::Online && app.active()));
            double seconds = live && !app.repaintTimer ? 1.0 / 60 : -1;
            if (!app.repaintTimer && !app.setup.message.empty()) {
                double messageDelay = app.setup.messageWait();
                seconds = seconds < 0 ? messageDelay : std::min(seconds, messageDelay);
            }
            if (app.resetAt) {
                double delay = std::max(.001, std::chrono::duration<double>(*app.resetAt - Clock::Source::now()).count());
                seconds = seconds < 0 ? delay : std::min(seconds, delay);
            }
            if (seconds > 0) glfwWaitEventsTimeout(seconds);
            else glfwWaitEvents();
        }
#ifdef _WIN32
        if (app.repaintTimer) KillTimer(glfwGetWin32Window(window), reinterpret_cast<UINT_PTR>(window));
#endif
        // detach callbacks before app is destroyed, ending their access to its stack address.
        glfwSetWindowRefreshCallback(window, nullptr);
        glfwSetFramebufferSizeCallback(window, nullptr);
        glfwSetWindowCloseCallback(window, nullptr);
        glfwSetCursorPosCallback(window, nullptr);
        glfwSetCursorEnterCallback(window, nullptr);
        glfwSetMouseButtonCallback(window, nullptr);
        glfwSetScrollCallback(window, nullptr);
        glfwSetCharCallback(window, nullptr);
        glfwSetKeyCallback(window, nullptr);
        glfwSetWindowUserPointer(window, nullptr);
        return 0;
    }
}

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
#else
int main() {
#endif
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
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
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
    int result = NEBULA::run(window);
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
