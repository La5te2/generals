// interface entry point: bring the game components together into an interactive desktop application.
// create the window and OpenGL context, handle page navigation, and route keyboard, mouse and console input.
// coordinate session controls and display updates, then release resources when the application closes.
#include "local.hpp"
#include "online.hpp"
#include "lan.hpp"
#include "playback.hpp"
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
        LanMatch lan;
        Playback replay;
        Human human; // keyboard and mouse controls for the player's selected cell and queued moves.
        Renderer renderer;
        Console console;
        Setup setup;
        MatchSnapshot displayed = local.snapshot();
        bool showingBoard = false, closing = false, quitRequested = false, leavingLan = false;
        unsigned int repaintTimer = 0; // native repaint interval in milliseconds. zero means the timer is stopped.
        bool painting = false;
        bool observingMatch = false;
        bool automatic = false;
        std::optional<Clock::Time> resetAt;
        int perspective = 2; // 0 = red, 1 = blue, 2 = full board.
        Tool hover = Tool::None;

        bool active() const { return displayed.active(); }
        bool running() const { return displayed.running(); }
        bool live() const { return quitRequested || leavingLan || (showingBoard && (running() || (setup.scene == Scene::Online && active()))); }
        // true while an unfinished match has a keyboard-and-mouse player, including a paused local match.
        bool hasHuman() const { return active() && human.player() >= 0; }
        PlayerInput& inputSession() {
            if (setup.scene != Scene::Online) return local;
            return setup.lan() ? static_cast<PlayerInput&>(lan) : online;
        }
        BoardControls controls() const { return {setup.scene, active(), running(), hasHuman(), hover, setup.scene == Scene::Replay && replay.cursor() > 0}; }
        const Observation& view() const { return (*displayed.views)[perspective]; }

        // Read the selected session and synchronize human controls with its latest observation.
        void update() {
            bool wasRunning = running();
            if (setup.scene == Scene::Replay) displayed = replay.snapshot();
            else if (setup.scene == Scene::Online) displayed = setup.lan() ? lan.snapshot() : online.snapshot();
            else displayed = local.snapshot();
            if (setup.scene == Scene::Online && displayed.player >= 0) {
                perspective = displayed.player;
                bool newGame = !wasRunning && running();
                if (setup.humanPlayer(Field::Player) && (human.player() < 0 || newGame) && view().cols > 0 && active()) human.reset(perspective, view());
            }
            human.sync(displayed);
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
            update();
            if (halfTurn == replay.length()) {
                report("Replay ended");
                scheduleReset();
            } else resetAt.reset();
        }

        // Advance playback, observe session completion, and finish pending navigation before drawing.
        bool refresh() {
            bool changed = false;
            if (!setup.message.empty() && setup.messageOpacity() == 0) { setup.message.clear(); changed = true; }
            if (leavingLan && !lan.opened()) {
                leavingLan = false;
                auto result = lan.snapshot();
                if (!result.error.empty()) setup.notify(result.error);
                if (!showingBoard) displayed = {};
                changed = true;
            }
            if (quitRequested && !lan.opened()) { finishQuit(); changed = true; }
            if (!showingBoard) return changed;
            if (setup.scene == Scene::Replay) {
                if (!replay.advance()) return changed;
                update();
                if (!active()) { report("Replay ended"); scheduleReset(); }
                return true;
            }
            auto previous = displayed;
            update();
            changed |= previous.views != displayed.views || previous.state != displayed.state || previous.status != displayed.status || previous.error != displayed.error;
            if (!previous.running() && running()) observingMatch = true;
            if (observingMatch && !active()) {
                observingMatch = false;
                auto result = view().result;
                report(setup.scene == Scene::Online ? displayed.status : result == Phases::RedWin ? "Red wins" : result == Phases::BlueWin ? "Blue wins" : "Draw");
                scheduleReset();
                hover = Tool::None;
                changed = true;
            }
            return changed;
        }

        // validate the selected mode before replacing the match and starting its timer.
        void start() {
            if (leavingLan || quitRequested) { setup.notify("Finishing the previous room"); return; }
            if (active()) { setup.notify("A match is already active"); return; }
            // consume a pending reset before starting, so a failed attempt stays available for manual retry.
            resetAt.reset();
            int player = -1;
            if (setup.scene == Scene::Online) {
                std::string command = setup.humanPlayer(Field::Player) ? "" : setup.field(Field::Player).input;
                if (setup.lan()) {
                    if (lan.opened()) { setup.notify("Finishing the previous room"); return; }
                    LanConfig config{setup.field(Field::Address).input, setup.field(Field::Username).input,
                        setup.field(Field::Room).input, command, setup.directory, setup.field(Field::Proxy).input};
                    if (!lan.start(config)) { setup.notify(lan.snapshot().error); return; }
                } else {
                    OnlineConfig config{setup.server == 1 ? Server::Main : Server::Bot,
                        setup.field(Field::Username).input, setup.field(Field::Identity).input,
                        command, setup.field(Field::Room).input, setup.field(Field::Proxy).input};
                    if (!online.start(config)) { setup.notify(online.snapshot().error); return; }
                }
            } else if (setup.scene == Scene::Replay) {
                const auto& name = setup.field(Field::Replay).input;
                std::string error;
                if (!replay.load(std::filesystem::path(std::u8string(name.begin(), name.end())), setup.milliseconds, error)) {
                    setup.notify(error);
                    return;
                }
            } else if (setup.scene == Scene::Local) {
                std::array<std::string, 2> commands{setup.field(Field::Red).input, setup.field(Field::Blue).input};
                if (setup.humanPlayer(Field::Red)) commands[0].clear();
                if (setup.humanPlayer(Field::Blue)) commands[1].clear();
                // One keyboard controls one player; LocalMatch itself permits externally supplied actions for both sides.
                if (commands[0].empty() && commands[1].empty()) { setup.notify("Choose a program for at least one player"); return; }
                player = commands[0].empty() ? 0 : commands[1].empty() ? 1 : -1;
                if (!local.start(commands, std::random_device{}(), setup.milliseconds, setup.directory)) {
                    update();
                    setup.notify(displayed.error);
                    return;
                }
            } else {
                setup.notify("Select a mode first");
                return;
            }

            // Enter the board only after the selected mode has accepted its configuration.
            setup.message.clear();
            setup.focus = setup.fileHover = Field::None;
            showingBoard = true;
            observingMatch = setup.scene != Scene::Replay;
            perspective = setup.scene == Scene::Online ? 0 : player >= 0 ? player : 2;
            hover = Tool::None;
            human.reset(-1, view());
            update();
            if (setup.scene == Scene::Local) human.reset(player, view());
            if (setup.scene == Scene::Replay && !active()) { report("Replay ended"); scheduleReset(); }
        }

        // stop keeps the final position on screen. reset starts another session with the current configuration.
        void stop() {
            if (!showingBoard) return;
            if (active()) {
                if (setup.scene == Scene::Online && setup.lan()) {
                    lan.stop(); human.deselect(); hover = Tool::None;
                    return; // the LAN worker reports completion after the room settles the surrender.
                }
                if (setup.scene == Scene::Replay) replay.stop();
                else if (setup.scene == Scene::Online) online.stop();
                else local.stop();
                update();
                report(setup.scene == Scene::Replay ? "Replay stopped" : "Match stopped");
                scheduleReset();
            }
            observingMatch = false;
            human.deselect();
            hover = Tool::None;
        }

        void reset() { if (showingBoard && controls().enabled(Tool::Reset)) start(); }

        void scheduleReset() {
            resetAt.reset();
            if (automatic && showingBoard && !active() && !closing && !quitRequested && displayed.error.empty()) {
                resetAt = Clock::Source::now() + std::chrono::milliseconds(1000);
            }
        }

        void setAuto(bool enabled) { automatic = enabled; scheduleReset(); }

        // return one page at a time: board to configuration, then configuration to home.
        void back() {
            if (showingBoard) {
                stop();
                if (setup.scene == Scene::Online && setup.lan()) { lan.leave(); leavingLan = lan.opened(); update(); }
                observingMatch = false;
                showingBoard = false;
                human.reset(-1, view());
            } else { setup.scene = Scene::Home; setup.message.clear(); }
            resetAt.reset();
            setup.focus = setup.fileHover = Field::None;
            console.opened = false;
            console.edit(Edit::Clear);
        }

        void quit() {
            stop();
            lan.leave();
            resetAt.reset();
            quitRequested = true;
            if (!lan.opened()) finishQuit();
        }

        void finishQuit() {
            quitRequested = false;
            observingMatch = false;
            std::string error;
            if (!local.savePending(setup.directory, error) || !lan.savePending(setup.directory, error)) {
                setup.notify(error);
                console.feedback = error;
                return;
            }
            closing = true;
        }

        // TURN <ms> sets the local half-turn duration and calls Playback::setInterval for replay speed.
        // ONLINE, including future LAN rooms, never reads this setting.
        bool setTurn(int milliseconds) {
            if (!local.setInterval(milliseconds)) return false;
            setup.milliseconds = milliseconds;
            replay.setInterval(milliseconds);
            return true;
        }

        // requested dimensions are content-area coordinates. two zeros select borderless full screen.
        bool setWindow(GLFWwindow* window, int requestedWidth, int requestedHeight) {
            bool fullscreen = requestedWidth == 0 && requestedHeight == 0;
            if (!fullscreen && (requestedWidth < minWindowWidth || requestedWidth > maxWindowWidth ||
                                requestedHeight < minWindowHeight || requestedHeight > maxWindowHeight)) return false;
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
            if (fullscreen) {
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
                int nextWidth = std::min(availableWidth, requestedWidth);
                int nextHeight = std::min(availableHeight, requestedHeight);
                glfwSetWindowSize(window, nextWidth, nextHeight);
                glfwSetWindowPos(window, left + frameLeft + (availableWidth - nextWidth) / 2,
                                 top + frameTop + (availableHeight - nextHeight) / 2);
            }
            hover = Tool::None;
            setup.fileHover = Field::None;
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
            if (command.type == Command::Rd) {
                std::filesystem::path directory = std::u8string(command.text.begin(), command.text.end());
                std::string error;
                if (!directory.empty() && !replayDirectory(directory, error)) { console.feedback = error; return; }
                if (!local.savePending(directory, error) || !lan.savePending(directory, error)) { console.feedback = error; return; }
                setup.directory = std::move(directory);
                if (showingBoard) update();
                console.feedback = command.text.empty() ? "Recording disabled for new matches" : "New matches record to: " + command.text;
                return;
            }
            if (command.type == Command::Win) {
                if (setWindow(window, command.value, command.height)) {
                    int width, height;
                    glfwGetWindowSize(window, &width, &height);
                    std::string label = command.value == 0 ? "Full screen" : "Window";
                    console.feedback = label + ": " + std::to_string(width) + " x " + std::to_string(height);
                } else console.feedback = "Window size update failed";
                return;
            }
            if (command.type == Command::Help) { console.feedback = Manual::commands; return; }
            if (command.type == Command::Man) { console.manual = true; return; }
            if (command.type == Command::Invalid) console.feedback = "Use help for commands or man for the game manual";
        }

        // board buttons and keyboard shortcuts control playback independently of console commands.
        void playback() {
            if (!showingBoard || !controls().enabled(Tool::Playback)) return;
            if (setup.scene == Scene::Replay) {
                replay.toggle();
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

        void browse(GLFWwindow* window, Field field) {
            std::string error;
            PathKind kind = setup.scene == Scene::Replay ? PathKind::Replay : PathKind::Program;
            bool program = kind == PathKind::Program;
            setup.fileHover = Field::None;
            auto selected = choosePath(window, kind, error);
            if (!selected) { if (!error.empty()) setup.notify(error); return; }
            std::error_code status;
            std::filesystem::path path(std::u8string(selected->begin(), selected->end()));
            bool valid = std::filesystem::is_regular_file(path, status);
            if (!valid) { setup.notify("Select an existing file"); return; }
            // only program command lines need quotes. the replay field holds the path itself.
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
            setup.field(field) = std::move(next);
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
                if (hasHuman()) {
                    human.click(inputSession(), displayed, x, y, width, height);
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
            for (Field field : inputOrder(setup)) {
                if (hasFileButton(field) && fileButton(setup, field, width, height).contains(x, y)) {
                    browse(window, field);
                    return;
                }
            }
            setup.focus = Field::None;
            if (setup.scene == Scene::Online) {
                for (int server = 0; server < 3; ++server) {
                    if (choiceButton(formControl(setup.scene, 0, width, height), server, 3).contains(x, y)) {
                        setup.selectServer(server);
                        return;
                    }
                }
            }
            for (Field field : inputOrder(setup)) {
                if (field != Field::None && inputField(setup, field, width, height).contains(x, y)) setup.focus = field;
            }
            if (setup.focus != Field::None) setup.field(setup.focus).edit(Edit::End);
            if (startButton(width, scale).contains(x, y)) start();
        }

        // keyboard text goes to the console while open, otherwise to the focused configuration field.
        TextInput* input() {
            if (console.opened) return &console;
            if (!showingBoard && setup.focus != Field::None) return &setup.field(setup.focus);
            return nullptr;
        }
    };

    void drawWindow(GLFWwindow* window);
    void redraw(GLFWwindow* window);

#ifdef _WIN32
    // timer callbacks also run while Windows handles a window drag inside event processing.
    void CALLBACK repaintWindow(HWND, UINT, UINT_PTR id, DWORD) {
        auto* window = reinterpret_cast<GLFWwindow*>(id);
        auto* app = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
        if (app) app->refresh();
        redraw(window);
    }
#endif

    // Schedule the next repaint. On Windows, callbacks invalidate the window and WM_PAINT performs the drawing.
    void redraw(GLFWwindow* window) {
#ifdef _WIN32
        auto* app = static_cast<WindowState*>(glfwGetWindowUserPointer(window));
        if (!app) return;
        HWND handle = glfwGetWin32Window(window);
        UINT_PTR id = reinterpret_cast<UINT_PTR>(window);
        bool live = app->live();
        if (live || !app->setup.message.empty()) {
            double delay = live ? 1.0 / 60 : app->setup.messageWait();
            UINT milliseconds = static_cast<UINT>(delay * 1000) + 1;
            // An unchanged interval keeps its original deadline when input requests another repaint.
            if (milliseconds != app->repaintTimer) {
                if (SetTimer(handle, id, milliseconds, repaintWindow)) app->repaintTimer = milliseconds;
                else { KillTimer(handle, id); app->repaintTimer = 0; }
            }
        } else if (app->repaintTimer) {
            KillTimer(handle, id);
            app->repaintTimer = 0;
        }
        RedrawWindow(handle, nullptr, nullptr, RDW_INVALIDATE | RDW_NOERASE);
#else
        drawWindow(window);
#endif
    }

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
            if (app.human.player() >= 0) queued = app.displayed.queued[app.human.player()];
            auto names = app.displayed.names;
            if (app.displayed.host >= 0) names[app.displayed.host] += " [HOST]";
            app.renderer.draw(app.view(), app.perspective, width, height, app.controls(), app.setup, app.console, app.human, queued, names, app.displayed.status);
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
        app.setWindow(window, 960, 800);
        // store app's address on the GLFW window so each callback can recover the same object.
        glfwSetWindowUserPointer(window, &app);
        // moving a window can hold execution inside GLFW's event handling.
        // refresh callbacks redraw there, while the outer loop waits for event handling to return.
        glfwSetWindowRefreshCallback(window, refreshWindow);
        glfwSetFramebufferSizeCallback(window, [](GLFWwindow* target, int, int) { redraw(target); });
        glfwSetWindowCloseCallback(window, [](GLFWwindow* target) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            state.quit();
            if (!state.closing) { glfwSetWindowShouldClose(target, GLFW_FALSE); redraw(target); }
        });
        // redraw a tooltip when the hovered tool changes, keeping ordinary mouse motion inexpensive.
        glfwSetCursorPosCallback(window, [](GLFWwindow* target, double x, double y) {
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            Tool hover = Tool::None;
            Field fileHover = Field::None;
            int width, height;
            glfwGetWindowSize(target, &width, &height);
            if (state.showingBoard && !state.console.opened) {
                for (Tool tool : tools) {
                    if (toolButton(tool, width, barScale(height)).contains(x, y)) hover = tool;
                }
            } else if (!state.showingBoard && !state.console.opened) {
                for (Field field : inputOrder(state.setup)) {
                    if (hasFileButton(field) && fileButton(state.setup, field, width, height).contains(x, y)) {
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
            state.setup.fileHover = Field::None;
            redraw(target);
        });
        // mouse events supply window coordinates, matching the layout functions in scene.hpp.
        glfwSetMouseButtonCallback(window, [](GLFWwindow* target, int button, int action, int) {
            if (action != GLFW_PRESS) return;
            auto& state = *static_cast<WindowState*>(glfwGetWindowUserPointer(target));
            if (button == GLFW_MOUSE_BUTTON_RIGHT && state.hasHuman() && !state.console.opened) {
                state.human.deselect();
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
                state.setup.fileHover = Field::None;
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
                    auto order = inputOrder(state.setup);
                    auto current = std::find(order.begin(), order.end(), state.setup.focus);
                    do {
                        if (current == order.end() || ++current == order.end()) current = order.begin();
                    } while (*current == Field::None && state.setup.scene != Scene::Home);
                    state.setup.focus = *current;
                    if (state.setup.focus != Field::None) state.setup.field(state.setup.focus).edit(Edit::End);
                } else if (auto* input = state.input()) {
                    editInput(target, *input, key, action, mods);
                    state.setup.notify(input->feedback);
                }
            } else if (state.hasHuman()) {
                state.refresh();
                if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) state.back();
                else if (key == GLFW_KEY_SPACE) { if (action == GLFW_PRESS) state.playback(); }
                else { state.human.key(state.inputSession(), state.displayed, key, action, mods); state.update(); }
            } else if (state.setup.scene == Scene::Replay && (key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT)) {
                state.useTool(key == GLFW_KEY_RIGHT ? Tool::Forward : Tool::Backward);
            } else if (action == GLFW_PRESS) {
                if (state.controls().spectator() && key >= GLFW_KEY_1 && key <= GLFW_KEY_3) {
                    state.perspective = key - GLFW_KEY_1;
                    state.update();
                }
                if (key == GLFW_KEY_SPACE) state.playback();
                if (key == GLFW_KEY_PERIOD) state.useTool(Tool::Forward);
                if (key == GLFW_KEY_COMMA) state.useTool(Tool::Backward);
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
            if (app.closing) break;
            if (app.resetAt && Clock::Source::now() >= *app.resetAt) {
                app.reset();
                repaint = true;
            }
            if (!app.repaintTimer && !app.setup.message.empty() && app.setup.messageOpacity() < 1) repaint = true;
            if (repaint) redraw(window);

            // wait for the earlier timer. a negative delay means that only input can wake the window.
            bool live = app.live();
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
