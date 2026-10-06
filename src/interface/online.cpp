// online session: keep server communication off the window thread, publish account-only views, and send one action per update.
// the server settles moves and decides the outcome. stopping a session cancels its queue entry or leaves its game.
#include "online.hpp"
#include "process.hpp"
#include <rtc/websocket.hpp>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <charconv>

namespace NEBULA::OnlineProtocol {
    using Json = nlohmann::json;
    inline constexpr std::int64_t maxInteger = 9007199254740991LL; // largest exact integer in the server's number format.
    inline constexpr std::string_view clientKey = "sd09fjdZ03i0ejwi_changeme"; // public web-client identifier.

    struct Error : std::runtime_error { using std::runtime_error::runtime_error; };

    inline void require(bool valid, const char* message) {
        if (!valid) throw Error(message);
    }

    // redact the account identifier before shortening errors from the transport or server.
    inline std::string diagnostic(std::string message, const std::string& userId) {
        if (!userId.empty()) {
            for (std::size_t pos = message.find(userId); pos != std::string::npos; pos = message.find(userId, pos + 10)) {
                message.replace(pos, userId.size(), "[redacted]");
            }
        }
        for (char& letter : message) {
            if (static_cast<unsigned char>(letter) < 32 || letter == 127) letter = ' ';
        }
        if (message.size() > 240) message = message.substr(0, 237) + "...";
        return message;
    }

    inline std::string rejection(std::string_view event, const Json& data, const std::string& userId) {
        std::string reason;
        if (event == "error_user_id") reason = "Account is active in another session";
        else if (event == "error_queue_full") reason = "Room is full";
        else if (event == "error_join_queue") reason = "Queue entry was rejected";
        else if (event == "error_banned") reason = "Account is temporarily banned";
        else if (event == "error_kicked" || event == "kicked") reason = "Account was removed from the room";
        else if (event == "server_down" || event == "server_restart") reason = "Server is restarting or unavailable";
        else if (event == "connect_error") reason = "Socket.IO connection was rejected";
        else reason = "Server rejected the request";
        reason += " (" + diagnostic(std::string(event), userId) + ")";
        if (data.is_string()) reason += ": " + diagnostic(data.get<std::string>(), userId);
        else if (data.is_object() && data.contains("message") && data["message"].is_string()) {
            reason += ": " + diagnostic(data["message"].get<std::string>(), userId);
        }
        return reason;
    }

    inline Json parse(std::string_view packet) {
        return Json::parse(packet, [](int depth, Json::parse_event_t, Json&) {
            require(depth < 32, "Server JSON exceeds its nesting limit");
            return true;
        });
    }

    inline std::int64_t integer(const Json& value, std::int64_t low, std::int64_t high) {
        require(value.is_number_integer(), "Expected an integer in server data");
        if (value.is_number_unsigned()) require(value.get<std::uint64_t>() <= static_cast<std::uint64_t>(high), "Server integer exceeds its range");
        auto number = value.get<std::int64_t>();
        require(number >= low && number <= high, "Server integer is outside its range");
        return number;
    }

    inline const Json& field(const Json& object, std::string_view name) {
        require(object.is_object() && object.contains(name), "Server packet is missing a required field");
        return object.at(name);
    }

    inline bool nickname(std::string_view expected, std::string_view actual) {
        if (expected == actual) return true;
        if (!expected.starts_with("[Bot]") || !actual.starts_with("[Bot]")) return false;
        expected.remove_prefix(5);
        actual.remove_prefix(5);
        while (expected.starts_with(' ')) expected.remove_prefix(1);
        while (actual.starts_with(' ')) actual.remove_prefix(1);
        return expected == actual;
    }

    inline void rules(const Json& data) {
        constexpr const char* exceeds = "Game rules exceed mainstream 1v1 rules";
        require(data.is_object(), "Expected a server game object");
        for (auto name : {"swamps", "deserts", "lookouts", "observatories", "tunnels", "strongholds"}) {
            if (data.contains(name)) require(data[name].is_null() || (data[name].is_array() && data[name].empty()), exceeds);
        }
        if (data.contains("options") && !data["options"].is_null()) {
            const auto& options = data["options"];
            require(options.is_object(), "Invalid game options");
            if (options.contains("modifiers")) require(options["modifiers"].is_array() && options["modifiers"].empty(), exceeds);
        }
    }

    // each pair copies an unchanged run, then replaces a run. the final unchanged run can stand alone.
    // build a new array so a malformed patch leaves the previous observation intact.
    inline std::vector<std::int64_t> patch(const std::vector<std::int64_t>& previous, const Json& diff, std::size_t limit) {
        require(diff.is_array() && diff.size() <= 3 * limit + 2, "Invalid map diff length");
        std::vector<std::int64_t> next;
        std::size_t source = 0, cursor = 0;
        while (cursor < diff.size()) {
            auto keep = static_cast<std::size_t>(integer(diff[cursor++], 0, static_cast<std::int64_t>(limit)));
            require(keep <= previous.size() - std::min(source, previous.size()) && keep <= limit - next.size(), "Map diff copies beyond its previous array");
            if (keep) next.insert(next.end(), previous.begin() + source, previous.begin() + source + keep);
            source += keep;
            if (cursor == diff.size()) break;
            auto replace = static_cast<std::size_t>(integer(diff[cursor++], 0, static_cast<std::int64_t>(limit)));
            require(replace <= diff.size() - cursor && replace <= limit - next.size(), "Map diff replacement is incomplete");
            for (std::size_t index = 0; index < replace; ++index) next.push_back(integer(diff[cursor++], -4, maxInteger));
            source += replace;
        }
        return next;
    }

    struct Board {
        std::vector<std::int64_t> map, cities;
        Observation view;
        bool received = false;

        // apply every diff, including repeated-turn updates. only a new turn requests another strategy action.
        bool update(const Json& data, int player) {
            rules(data);
            auto tick = static_cast<std::uint64_t>(integer(field(data, "turn"), 1, maxInteger));
            require(!received || tick >= view.tick, "Server turn moved backwards");
            auto nextMap = patch(map, field(data, "map_diff"), 2 + 3 * Capacity);
            auto nextCities = patch(cities, field(data, "cities_diff"), Capacity);
            require(nextMap.size() >= 2, "Server map has no dimensions");
            auto width = nextMap[0], height = nextMap[1];
            require(width >= 1 && width <= dim && height >= 1 && height <= dim, "Server map dimensions exceed board capacity");
            int count = static_cast<int>(width * height);
            require(nextMap.size() == static_cast<std::size_t>(2 + 2 * count) || nextMap.size() == static_cast<std::size_t>(2 + 3 * count), "Unexpected server map layers");
            if (received) require(width == view.cols && height == view.rows, "Map dimensions changed during the game");
            // current servers include a tunnel layer even on ordinary maps. retain it when applying later diffs.
            for (std::size_t cell = 2 + 2 * count; cell < nextMap.size(); ++cell) require(nextMap[cell] == 0, "Tunnel maps require additional rules");
            Observation next;
            next.cols = static_cast<int>(width);
            next.rows = static_cast<int>(height);
            next.player = player;
            next.tick = tick;
            for (int cell = 0; cell < count; ++cell) {
                auto terrain = nextMap[2 + count + cell], army = nextMap[2 + cell];
                require(terrain >= -4 && terrain <= 1, "Server map contains an unsupported player or terrain");
                auto& shown = next.cells[cell];
                if (terrain == -3 || terrain == -4) {
                    shown.terrain = terrain == -3 ? ViewTerrain::Fog : ViewTerrain::Obstacle;
                    continue;
                }
                require(army >= 0, "Visible army is negative");
                shown.terrain = terrain == -2 ? ViewTerrain::Mountain : ViewTerrain::Plain;
                shown.owner = terrain >= 0 ? static_cast<std::int8_t>(terrain) : -1;
                shown.army = army;
            }
            std::array<bool, Capacity> city{};
            for (auto cell : nextCities) {
                require(cell >= 0 && cell < count && !city[cell], "Invalid or duplicate city index");
                city[cell] = true;
                auto& shown = next.cells[cell];
                if (shown.terrain == ViewTerrain::Fog || shown.terrain == ViewTerrain::Obstacle) continue;
                require(shown.terrain == ViewTerrain::Plain, "City conflicts with terrain");
                shown.terrain = ViewTerrain::City;
            }
            const auto& generals = field(data, "generals");
            require(generals.is_array() && generals.size() == 2, "A game requires two general entries");
            for (const auto& entry : generals) {
                auto cell = integer(entry, -1, count - 1);
                if (cell < 0) continue;
                auto& shown = next.cells[cell];
                if (shown.terrain == ViewTerrain::Fog || shown.terrain == ViewTerrain::Obstacle) continue;
                require(shown.terrain == ViewTerrain::Plain && shown.owner >= 0, "General conflicts with terrain");
                shown.terrain = ViewTerrain::General;
            }
            const auto& scores = field(data, "scores");
            require(scores.is_array() && scores.size() == 2, "A game requires two score entries");
            std::array<bool, 2> seen{};
            for (const auto& score : scores) {
                int side = static_cast<int>(integer(field(score, "i"), 0, 1));
                require(!seen[side], "Duplicate player score");
                seen[side] = true;
                next.land[side] = static_cast<int>(integer(field(score, "tiles"), 0, count));
                next.armies[side] = integer(field(score, "total"), 0, maxInteger);
            }
            bool advanced = !received || tick > view.tick;
            map = std::move(nextMap);
            cities = std::move(nextCities);
            view = std::move(next);
            received = true;
            return advanced;
        }
    };

    inline std::optional<int> target(const Observation& view, const Action& action) {
        if (action.type != ActionType::Move || action.row < 0 || action.row >= view.rows || action.col < 0 || action.col >= view.cols) return {};
        int row = action.row, col = action.col;
        switch (action.direction) {
            case Direction::Up: --row; break;
            case Direction::Down: ++row; break;
            case Direction::Left: --col; break;
            case Direction::Right: ++col; break;
            default: return {};
        }
        if (row < 0 || row >= view.rows || col < 0 || col >= view.cols) return {};
        int cell = row * view.cols + col;
        if (view.cells[cell].terrain == ViewTerrain::Mountain) return {};
        return cell;
    }

    inline Json join(const OnlineConfig& config) {
        if (!config.room.empty()) {
            Json event = Json::array({"join_private", config.room, config.userId});
            if (config.server == Server::Main) { event.push_back(clientKey); event.push_back(nullptr); }
            return event;
        }
        // the final flag allows Bot opponents. the local input source leaves account and matchmaking options unchanged.
        if (config.server == Server::Main) return Json::array({"join_1v1", config.userId, clientKey, 0, nullptr, true});
        return Json::array({"join_1v1", config.userId, nullptr, 0});
    }
}

namespace NEBULA {
    using namespace OnlineProtocol;
    using Time = std::chrono::steady_clock;
    using namespace std::chrono_literals;

    struct OnlineMatch::Session {
        struct Frame { std::string text; Time::time_point arrival; };
        // callbacks retain this mailbox instead of the session. old callbacks can only touch their old connection.
        struct Mailbox {
            std::mutex mutex;
            std::condition_variable changed;
            std::deque<Frame> frames;
            std::deque<Action> inputs;
            std::optional<Action> sent;
            std::size_t bytes = 0;
            bool closing = false, human = true;
            std::string failure;
            MatchSnapshot published;
        };
        std::shared_ptr<Mailbox> mail = std::make_shared<Mailbox>();
        std::thread worker;

        void stop() {
            {
                std::lock_guard lock(mail->mutex);
                mail->closing = true;
                mail->inputs.clear();
                mail->sent.reset();
                mail->changed.notify_all();
            }
            if (worker.joinable()) worker.join();
        }

        void run(OnlineConfig config) {
            rtc::WebSocketConfiguration transport;
            transport.connectionTimeout = 10s;
            transport.maxMessageSize = 256 * 1024;
            // an empty proxy selects a direct connection. both routes retain the library's TLS verification.
            if (!config.proxy.empty()) transport.proxyServer = rtc::ProxyServer(config.proxy);
            rtc::WebSocket socket(transport);
            auto inbox = mail;
            auto fail = [inbox](std::string message) {
                std::lock_guard lock(inbox->mutex);
                if (inbox->failure.empty()) inbox->failure = std::move(message);
                inbox->changed.notify_all();
            };
            socket.onMessage([inbox, fail](rtc::message_variant message) {
                auto text = std::get_if<std::string>(&message);
                if (!text) { fail("Unexpected binary server packet"); return; }
                std::lock_guard lock(inbox->mutex);
                if (inbox->closing || !inbox->failure.empty()) return;
                // diffs depend on all preceding packets. overflow ends the session instead of dropping a packet.
                if (text->size() > 256 * 1024 || inbox->frames.size() >= 128 || inbox->bytes + text->size() > 2 * 1024 * 1024) {
                    inbox->failure = "Server receive queue exceeded its limit";
                } else {
                    inbox->bytes += text->size();
                    inbox->frames.push_back({std::move(*text), Time::now()});
                }
                inbox->changed.notify_all();
            });
            socket.onError([fail, userId = config.userId](std::string reason) {
                std::string message = "WebSocket connection failed";
                if (!reason.empty()) message += ": " + diagnostic(std::move(reason), userId);
                fail(std::move(message));
            });
            socket.onClosed([fail] { fail("Server connection closed"); });

            MatchSnapshot state;
            state.state = MatchState::Active;
            state.running = true;
            state.status = "Connecting";
            state.names = {"", ""};
            OnlineProtocol::Board board;
            StrategyProcess strategy;
            bool opened = false, connected = false, joined = false, started = false;
            bool strategyStarted = false, acted = true, completed = false;
            auto handshakeDeadline = Time::now() + 15s;
            auto heartbeatDeadline = handshakeDeadline, queueDeadline = Time::time_point::max();
            auto gameDeadline = Time::time_point::max(), actionDeadline = Time::time_point::min();
            auto heartbeat = 45s;
            Time::time_point previousUpdate{};
            std::chrono::milliseconds cadence{500}; // initial update interval estimate until consecutive ticks arrive.

            auto publish = [&] {
                std::lock_guard lock(mail->mutex);
                state.queued = {};
                if (state.player >= 0) {
                    auto& queue = state.queued[state.player];
                    if (mail->sent) queue.push_back(*mail->sent);
                    queue.insert(queue.end(), mail->inputs.begin(), mail->inputs.end());
                }
                mail->published = state;
            };
            auto publishView = [&] {
                auto views = std::make_shared<std::array<Observation, 3>>();
                (*views)[state.player] = board.view;
                state.views = std::move(views);
                publish();
            };
            auto send = [&](std::string packet) {
                require(socket.isOpen(), "Server connection is closed");
                socket.send(std::move(packet));
            };
            auto emit = [&](const Json& event) { send("42" + event.dump()); };
            auto joinQueue = [&] {
                {
                    std::lock_guard lock(mail->mutex);
                    if (mail->closing) return;
                }
                if (config.server == Server::Main) emit(Json::array({"stars_and_rank", config.userId}));
                else emit(Json::array({"stars_and_rank", config.userId, nullptr}));
                emit(join(config));
                joined = true;
                state.status = config.room.empty() ? "Waiting for 1v1" : "Joining private room";
                // ranked queues can stay silent until matched. private rooms acknowledge entry with queue_update.
                queueDeadline = config.room.empty() ? Time::time_point::max() : Time::now() + 30s;
                publish();
            };
            auto publicName = [&](const Json& value) {
                require(value.is_string(), "Server username must be text");
                std::string name = value.get<std::string>();
                require(name.size() <= 256, "Server username exceeds its size limit");
                for (std::size_t pos = name.find(config.userId); pos != std::string::npos; pos = name.find(config.userId, pos + 10)) name.replace(pos, config.userId.size(), "[redacted]");
                name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char ch) { return ch < 32 || ch == 127; }), name.end());
                return name;
            };
            auto event = [&](const Json& packet) {
                require(packet.is_array() && !packet.empty() && packet[0].is_string(), "Invalid server event");
                auto name = packet[0].get<std::string>();
                const Json data = packet.size() > 1 ? packet[1] : Json{};
                if (name.starts_with("error") || name == "gio_error" || name == "game_error" || name == "kicked" || name == "server_down" || name == "server_restart") {
                    throw Error(rejection(name, data, config.userId));
                }
                if (name == "queue_left" || name == "removed_from_queue") {
                    require(started, "Server removed the queue entry");
                } else if (name == "queue_update" && joined && !started) {
                    require(data.is_object(), "Invalid queue update");
                    if (data.contains("id") && data["id"] != (config.room.empty() ? "duel" : config.room)) return;
                    queueDeadline = Time::time_point::max();
                    state.status = config.room.empty() ? "Waiting for 1v1" : "In private room";
                    if (data.contains("numPlayers")) state.status += " (" + std::to_string(integer(data["numPlayers"], 0, 64)) + ")";
                    publish();
                } else if (name == "pre_game_start" && joined && !started) {
                    state.status = "Starting game";
                    queueDeadline = Time::now() + 30s;
                    publish();
                } else if (name == "game_start") {
                    require(joined && !started, "Unexpected game start");
                    rules(data);
                    const auto& names = field(data, "usernames");
                    require(names.is_array() && names.size() == 2, "Online mode requires exactly two players");
                    int player = static_cast<int>(integer(field(data, "playerIndex"), 0, 1));
                    require(names[player].is_string() && nickname(config.username, names[player].get<std::string>()), "Username differs from this account. Check username and User ID");
                    if (data.contains("teams") && !data["teams"].is_null()) {
                        const auto& teams = data["teams"];
                        require(teams.is_array() && teams.size() == 2 && teams[0] != teams[1], "Online mode requires opposing teams");
                    }
                    if (config.room.empty()) require(field(data, "game_type") == "1v1", "Server returned a different game mode");
                    started = true;
                    state.player = player;
                    state.names = {publicName(names[0]), publicName(names[1])};
                    state.status = "Waiting for board";
                    queueDeadline = Time::time_point::max();
                    gameDeadline = Time::now() + 30s;
                    publish();
                } else if (name == "game_update") {
                    require(started, "Board update arrived before game start");
                    bool advanced = board.update(data, state.player);
                    if (advanced) {
                        std::lock_guard lock(mail->mutex);
                        mail->sent.reset();
                    }
                    gameDeadline = Time::now() + 30s;
                    state.status = "Playing";
                    publishView();
                    if (!advanced) return;
                    acted = false;
                    if (!config.command.empty()) {
                        if (!strategyStarted) {
                            require(strategy.start(config.command, {state.player, board.view.rows, board.view.cols}), "Strategy process could not start");
                            strategyStarted = true;
                        }
                        strategy.request(board.view);
                    }
                } else if ((name == "game_won" || name == "game_lost") && started) {
                    int winner = name == "game_won" ? state.player : 1 - state.player;
                    board.view.result = winner == 0 ? Phases::RedWin : Phases::BlueWin;
                    state.status = name == "game_won" ? "Victory" : "Defeat";
                    completed = true;
                    publishView();
                }
            };

            try {
                socket.open(config.server == Server::Main ? "wss://ws.generals.io/socket.io/?EIO=4&transport=websocket" : "wss://botws.generals.io/socket.io/?EIO=4&transport=websocket");
                while (!completed) {
                    std::deque<Frame> frames;
                    {
                        std::unique_lock lock(mail->mutex);
                        mail->changed.wait_for(lock, strategyStarted || started ? 5ms : 100ms, [&] {
                            return mail->closing || !mail->frames.empty() || !mail->failure.empty();
                        });
                        if (mail->closing) break;
                        frames.swap(mail->frames);
                        mail->bytes = 0;
                        // consume final result packets before handling a transport close in the same batch.
                        if (frames.empty() && !mail->failure.empty()) throw Error(mail->failure);
                    }
                    for (const auto& frame : frames) {
                        {
                            std::lock_guard lock(mail->mutex);
                            if (mail->closing) break;
                        }
                        const auto& packet = frame.text;
                        require(!packet.empty(), "Empty server packet");
                        if (packet.front() == '0') {
                            require(!opened, "Repeated transport handshake");
                            auto hello = parse(std::string_view(packet).substr(1));
                            require(field(hello, "sid").is_string(), "Invalid transport session identifier");
                            auto interval = integer(field(hello, "pingInterval"), 1, 120000);
                            auto timeout = integer(field(hello, "pingTimeout"), 1, 120000);
                            heartbeat = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::milliseconds(interval + timeout)) + 1s;
                            heartbeatDeadline = frame.arrival + heartbeat;
                            opened = true;
                            state.status = "Connecting to game service";
                            publish();
                            send("40");
                        } else if (packet.front() == '2') {
                            require(opened, "Heartbeat arrived before handshake");
                            heartbeatDeadline = frame.arrival + heartbeat;
                            send("3" + packet.substr(1));
                        } else if (packet.starts_with("40")) {
                            require(opened && !connected, "Unexpected connection acknowledgement");
                            auto welcome = parse(std::string_view(packet).substr(2));
                            require(field(welcome, "sid").is_string(), "Invalid connection acknowledgement");
                            connected = true;
                            if (config.room.empty()) { send("420[\"queue_count\"]"); handshakeDeadline = Time::now() + 5s; }
                            else joinQueue();
                        } else if (packet.starts_with("430") && connected && !joined) {
                            require(parse(std::string_view(packet).substr(3)).is_array(), "Invalid queue acknowledgement");
                            joinQueue();
                        } else if (packet.starts_with("42")) {
                            require(connected, "Game event arrived before connection acknowledgement");
                            // events may ask for an acknowledgement. preserve its numeric identifier.
                            std::size_t cursor = 2;
                            while (cursor < packet.size() && packet[cursor] >= '0' && packet[cursor] <= '9') ++cursor;
                            require(cursor - 2 <= 12, "Oversized event acknowledgement identifier");
                            auto payload = parse(std::string_view(packet).substr(cursor));
                            bool update = payload.is_array() && !payload.empty() && payload[0] == "game_update";
                            auto oldTick = board.view.tick;
                            event(payload);
                            if (update && board.view.tick > oldTick) {
                                // consecutive ticks estimate the update interval, clamped to 50 through 5000 ms.
                                // accept actions during the first 80% to leave time for transmission before the next update.
                                // this is a client-side cutoff. the server determines when each action executes.
                                if (previousUpdate != Time::time_point{} && board.view.tick - oldTick == 1) {
                                    cadence = std::clamp(std::chrono::duration_cast<std::chrono::milliseconds>(frame.arrival - previousUpdate), 50ms, 5000ms);
                                }
                                previousUpdate = frame.arrival;
                                actionDeadline = frame.arrival + cadence * 4 / 5;
                            }
                            if (cursor > 2) send("43" + packet.substr(2, cursor - 2) + "[]");
                        } else if (packet.starts_with("44")) {
                            auto reason = parse(std::string_view(packet).substr(2));
                            throw Error(rejection("connect_error", reason, config.userId));
                        } else if (packet == "1") {
                            throw Error("Server closed the Engine.IO transport");
                        } else if (packet.starts_with("41")) {
                            throw Error("Server disconnected the Socket.IO session");
                        } else require(packet == "6" || packet.front() == '3' || packet.starts_with("43"), "Unsupported server packet");
                        if (completed) break;
                    }
                    if (completed) break;
                    auto now = Time::now();
                    require(joined || now < handshakeDeadline, "Connection or queue handshake timed out");
                    require(now < heartbeatDeadline, "Server heartbeat timed out");
                    require(now < queueDeadline, "Private room acknowledgement timed out");
                    require(now < gameDeadline, "Server board updates timed out");
                    if (strategyStarted) require(strategy.error().empty(), "Strategy stopped or returned an invalid action");
                    if (!board.received || acted) continue;
                    if (now >= actionDeadline) { acted = true; continue; }
                    std::optional<Action> action;
                    {
                        std::lock_guard lock(mail->mutex);
                        if (mail->closing || !mail->frames.empty()) continue;
                        if (!strategyStarted && !mail->inputs.empty()) {
                            action = mail->inputs.front();
                            mail->inputs.pop_front();
                        }
                    }
                    if (strategyStarted) action = strategy.reply(board.view.tick, actionDeadline);
                    if (!action) continue;
                    acted = true;
                    if (auto to = target(board.view, *action)) {
                        int from = action->row * board.view.cols + action->col;
                        const auto& cell = board.view.cells[from];
                        if (cell.owner == state.player && cell.army >= 2) {
                            std::lock_guard lock(mail->mutex);
                            if (mail->closing || !mail->frames.empty()) continue;
                            // keep at most one move on the server. the remaining human route stays locally cancellable.
                            emit(Json::array({"attack", from, *to, action->half}));
                            if (!strategyStarted) mail->sent = *action;
                        }
                    }
                    publish();
                }
            } catch (const Error& error) { state.error = error.what(); }
            catch (const Json::exception&) { state.error = "Malformed JSON in server data"; }
            catch (const std::exception&) { state.error = "Online transport or strategy operation failed"; }
            if (!state.error.empty()) state.error = state.status + ": " + state.error;

            // leave before closing the socket. failure here still releases the connection and its strategy process.
            try {
                if (joined && socket.isOpen()) emit(Json::array({started ? "leave_game" : "cancel"}));
                if (socket.isOpen()) { send("41"); socket.close(); }
            } catch (const std::exception&) { socket.forceClose(); }
            strategy.stop();
            socket.resetCallbacks();
            socket.forceClose();
            state.state = MatchState::Finished;
            state.running = false;
            if (!state.error.empty()) state.status = "Connection ended";
            else if (!completed) state.status = "Stopped";
            {
                std::lock_guard lock(mail->mutex);
                mail->inputs.clear();
                mail->sent.reset();
                mail->closing = true;
            }
            publish();
        }
    };

    OnlineMatch::OnlineMatch() : session(std::make_unique<Session>()) {}
    OnlineMatch::~OnlineMatch() { stop(); }

    bool OnlineMatch::start(const OnlineConfig& config) {
        if (snapshot().state == MatchState::Active) return false;
        auto text = [](std::string_view value, std::size_t maximum) {
            return !value.empty() && value.size() <= maximum &&
                std::all_of(value.begin(), value.end(), [](unsigned char ch) { return ch >= 32 && ch != 127; });
        };
        std::string error;
        if (config.server != Server::Main && config.server != Server::Bot) error = "Choose Main or Bot server";
        else if (!text(config.username, 256) || !text(config.userId, 512) || config.userId.find(' ') != std::string::npos) error = "Enter a username and a single-line User ID";
        else if (config.username.find(config.userId) != std::string::npos) error = "Username and User ID must be separate values";
        else if (config.room.size() > 128 || !std::all_of(config.room.begin(), config.room.end(), [](unsigned char ch) {
            return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
        })) error = "Room requires an ID containing letters, digits, underscores or hyphens";
        else if (!config.command.empty() && !StrategyProcess::available(config.command)) error = "Strategy program was not found or its command is invalid";
        // this client accepts anonymous HTTP CONNECT endpoints with an explicit host and port.
        if (error.empty() && !config.proxy.empty()) {
            try {
                require(config.proxy.starts_with("http://") && config.proxy.size() <= 2048 &&
                    config.proxy.find_first_of("/?#@ \t\r\n", 7) == std::string::npos, "Proxy requires http://host:port");
                rtc::ProxyServer proxy(config.proxy);
                auto colon = config.proxy.rfind(':');
                require(colon > 6 && colon + 1 < config.proxy.size(), "Proxy requires an explicit port");
                int port = 0;
                const char* end = config.proxy.data() + config.proxy.size();
                auto parsed = std::from_chars(config.proxy.data() + colon + 1, end, port);
                require(parsed.ec == std::errc{} && parsed.ptr == end && port >= 1 && port <= 65535, "Proxy port is outside its range");
                require(proxy.type == rtc::ProxyServer::Type::Http && !proxy.hostname.empty() && proxy.port > 0 &&
                    !proxy.username && !proxy.password, "Proxy requires an HTTP endpoint without authentication");
            } catch (const std::exception&) { error = "Proxy requires http://host:port without authentication"; }
        }
        if (!error.empty()) {
            std::lock_guard lock(session->mail->mutex);
            session->mail->published.error = std::move(error);
            return false;
        }
        stop();
        session = std::make_unique<Session>();
        session->mail->human = config.command.empty();
        auto& published = session->mail->published;
        published.state = MatchState::Active;
        published.running = true;
        published.status = "Connecting";
        published.names = {"", ""};
        try {
            session->worker = std::thread([state = session.get(), config] {
                try { state->run(config); }
                catch (const std::exception&) {
                    std::lock_guard lock(state->mail->mutex);
                    state->mail->closing = true;
                    state->mail->published.state = MatchState::Finished;
                    state->mail->published.running = false;
                    state->mail->published.error = "Online session initialization failed";
                }
            });
        } catch (const std::exception&) {
            published.state = MatchState::Finished;
            published.running = false;
            published.error = "Online worker could not start";
            return false;
        }
        return true;
    }

    void OnlineMatch::stop() { session->stop(); }

    MatchSnapshot OnlineMatch::snapshot() const {
        std::lock_guard lock(session->mail->mutex);
        return session->mail->published;
    }

    bool OnlineMatch::enqueue(int player, const Action& action) {
        auto& mail = *session->mail;
        std::lock_guard lock(mail.mutex);
        const auto& shown = mail.published;
        if (mail.closing || !mail.human || shown.state != MatchState::Active || player < 0 || player != shown.player || mail.inputs.size() >= Capacity) return false;
        if (!target((*shown.views)[player], action)) return false;
        mail.inputs.push_back(action);
        auto& queue = mail.published.queued[player];
        queue.clear();
        if (mail.sent) queue.push_back(*mail.sent);
        queue.insert(queue.end(), mail.inputs.begin(), mail.inputs.end());
        mail.changed.notify_all();
        return true;
    }

    std::optional<Action> OnlineMatch::cancel(int player, bool all) {
        auto& mail = *session->mail;
        std::lock_guard lock(mail.mutex);
        if (player < 0 || player != mail.published.player || mail.inputs.empty()) return {};
        Action removed = all ? mail.inputs.front() : mail.inputs.back();
        if (all) mail.inputs.clear();
        else mail.inputs.pop_back();
        auto& queue = mail.published.queued[player];
        queue.clear();
        if (mail.sent) queue.push_back(*mail.sent);
        queue.insert(queue.end(), mail.inputs.begin(), mail.inputs.end());
        return removed;
    }
}
