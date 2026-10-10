// LAN matchmaking and play: a player-hosted WebSocket lobby routes clients to independent rooms.
// each room owns a local match; its creator owns the room lifetime, not the lobby lifetime.
#include "lan.hpp"
#include "local.hpp"
#include "proxy.hpp"
#include <rtc/websocket.hpp>
#include <rtc/websocketserver.hpp>
#include <nlohmann/json.hpp>
#include <atomic>
#include <charconv>
#include <deque>
#include <future>
#include <random>
#include <regex>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace NEBULA {
    namespace {
        using Json = nlohmann::json;
        using Socket = std::shared_ptr<rtc::WebSocket>;
        using Time = std::chrono::steady_clock;
        using namespace std::chrono_literals;
        constexpr std::size_t packetLimit = 256 * 1024, recordingLimit = 16 * 1024 * 1024;

        void require(bool value, const char* error) { if (!value) throw std::runtime_error(error); }
        bool nameValid(const std::string& value) {
            return !value.empty() && value.size() <= 64 &&
                std::all_of(value.begin(), value.end(), [](unsigned char c) { return c >= 32 && c <= 126; });
        }
        struct Address { std::string host, url; std::uint16_t port; bool listen; };
        Address address(std::string text) {
            if (text.empty()) text = "0.0.0.0:8080";
            if (text.find("://") == std::string::npos) text = "ws://" + text;
            std::smatch parts;
            static const std::regex pattern(R"(^(ws|wss)://(\[[0-9a-fA-F:]+\]|[A-Za-z0-9_.-]+)(?::([0-9]+))?(/[^\s#?]*)?$)");
            require(std::regex_match(text, parts, pattern), "Use IP:port or ws://host:port/path");
            bool secure = parts[1] == "wss";
            unsigned port = secure ? 443 : 8080;
            if (parts[3].matched) {
                auto number = parts[3].str();
                auto [end, error] = std::from_chars(number.data(), number.data() + number.size(), port);
                require(error == std::errc{} && end == number.data() + number.size() && port > 0 && port <= 65535,
                        "LAN port must be between 1 and 65535");
            }
            std::string host = parts[2];
            auto target = host == "0.0.0.0" ? "127.0.0.1" : host == "[::]" ? "[::1]" : host;
            Address result{host, std::string(secure ? "wss://" : "ws://") + target + ":" + std::to_string(port) + parts[4].str(),
                           static_cast<std::uint16_t>(port), !secure && (!parts[4].matched || parts[4] == "/")};
            if (host.front() == '[') result.host = host.substr(1, host.size() - 2);
            return result;
        }
        // SO_REUSEADDR permits duplicate listeners on Windows. the worker holds this lock
        // throughout the listener lifetime so simultaneous launches cannot both become the lobby.
        struct Listener {
#ifdef _WIN32
            HANDLE handle = nullptr;
            bool held = false;
            explicit Listener(unsigned port) {
                auto name = L"Local\\Generals.LAN." + std::to_wstring(port);
                handle = CreateMutexW(nullptr, FALSE, name.c_str());
                require(handle != nullptr, "Cannot reserve LAN listener");
                auto result = WaitForSingleObject(handle, 0);
                held = result == WAIT_OBJECT_0 || result == WAIT_ABANDONED;
            }
            ~Listener() { if (held) ReleaseMutex(handle); if (handle) CloseHandle(handle); }
            bool available() const { return held; }
#else
            explicit Listener(unsigned) {}
            bool available() const { return true; }
#endif
        };
        std::string encode(const Action& action) {
            std::ostringstream stream;
            require(Protocol::writeAction(stream, action), "Invalid LAN action");
            return stream.str();
        }
        Action decode(const Json& text) {
            require(text.is_string() && text.get_ref<const std::string&>().size() <= 100, "Invalid LAN action");
            std::istringstream stream(text.get<std::string>());
            auto action = Protocol::readAction(stream);
            stream >> std::ws;
            require(action.has_value() && stream.eof(), "Invalid LAN action");
            return *action;
        }
        struct Packet { Socket socket; std::string text; };
        struct Mail {
            std::mutex mutex;
            std::deque<Packet> packets;
            std::deque<Socket> accepted;
            std::deque<Json> commands;
            std::size_t bytes = 0;
            std::atomic<bool> quit{false}, leaving{false};
        };
        void listen(const Socket& socket, const std::shared_ptr<Mail>& mail) {
            std::weak_ptr<rtc::WebSocket> weak = socket;
            socket->onMessage([mail, weak](rtc::message_variant message) {
                auto source = weak.lock();
                if (!source || mail->quit) return;
                auto text = std::get_if<std::string>(&message);
                bool overflow;
                {
                    std::lock_guard lock(mail->mutex);
                    if (mail->quit) return;
                    overflow = !text || text->size() > packetLimit || mail->packets.size() >= 512 ||
                               mail->bytes + (text ? text->size() : 0) > 4 * 1024 * 1024;
                    if (!overflow) { mail->bytes += text->size(); mail->packets.push_back({source, std::move(*text)}); }
                }
                if (overflow) source->forceClose();
            });
        }
        void send(const Socket& socket, const Json& message) {
            require(socket && socket->isOpen(), "LAN connection closed");
            require(socket->bufferedAmount() < 2 * 1024 * 1024, "LAN connection is not receiving updates");
            socket->send(message.dump());
        }
        Json parse(const std::string& text) {
            auto value = Json::parse(text, [](int depth, Json::parse_event_t, Json&) {
                require(depth <= 12, "LAN packet nesting is too deep");
                return true;
            });
            require(value.is_object(), "Expected LAN packet object");
            return value;
        }
        Socket connect(const Address& endpoint, const std::shared_ptr<Mail>& mail, std::chrono::milliseconds timeout,
                       const std::optional<rtc::ProxyServer>& proxy) {
            rtc::WebSocket::Configuration options;
            options.proxyServer = proxy;
            options.connectionTimeout = timeout;
            options.pingInterval = 5s;
            options.maxOutstandingPings = 3;
            options.maxMessageSize = packetLimit;
            auto socket = std::make_shared<rtc::WebSocket>(options);
            listen(socket, mail);
            socket->open(endpoint.url);
            auto deadline = Time::now() + timeout;
            while (!socket->isOpen() && !socket->isClosed() && Time::now() < deadline && !mail->leaving)
                std::this_thread::sleep_for(10ms);
            if (socket->isOpen()) return socket;
            socket->forceClose();
            return {};
        }
    }

    struct LanMatch::Lobby {
        struct Client {
            Socket socket;
            std::string name, room;
            int seat = -1;
            bool ready = true, leaving = false;
            std::uint64_t ack = 0, game = 0, recorded = 0, revision = 0;
            std::size_t offset = 0;
            Time::time_point last = Time::now();
            std::shared_ptr<const std::array<Observation, 3>> views;
        };
        struct Room {
            std::string key, label;
            bool publicRoom = false;
            std::array<std::shared_ptr<Client>, 2> players;
            std::array<std::string, 2> names{"RED", "BLUE"};
            LocalMatch match;
            int milliseconds = 500;
            std::uint64_t game = 0, revision = 1;
            std::string replay;
            bool finished = false;
            std::optional<Time::time_point> closing;
        };
        std::shared_ptr<Mail> mail = std::make_shared<Mail>();
        std::thread worker;
        bool opened = false;

        explicit Lobby(Address endpoint) {
            std::promise<bool> ready;
            auto result = ready.get_future();
            worker = std::thread([this, endpoint, ready = std::move(ready)]() mutable { run(endpoint, ready); });
            opened = result.get();
        }
        ~Lobby() { mail->leaving = true; if (worker.joinable()) worker.join(); }
        void run(const Address& endpoint, std::promise<bool>& ready) {
            std::vector<std::shared_ptr<Client>> clients;
            std::vector<std::unique_ptr<Room>> rooms;
            std::uint64_t number = 0;
            bool announced = false;
            auto finish = [](Room& room, int loser) {
                room.match.stop(loser);
                for (auto& player : room.players) if (player) player->ready = false;
                ++room.revision;
            };
            auto detach = [&](Room& room, int seat) {
                finish(room, seat);
                if (seat == 0) room.closing = Time::now();
                room.players[seat].reset();
            };
            try {
                Listener reservation(endpoint.port);
                require(reservation.available(), "LAN listener already reserved");
                rtc::WebSocketServer::Configuration options;
                options.port = endpoint.port; options.bindAddress = endpoint.host;
                options.connectionTimeout = 5s; options.maxMessageSize = packetLimit;
                rtc::WebSocketServer server(options);
                server.onClient([inbox = mail](Socket socket) {
                    bool full;
                    {
                        std::lock_guard lock(inbox->mutex);
                        full = inbox->quit || inbox->accepted.size() >= 16;
                        if (!full) inbox->accepted.push_back(socket);
                    }
                    if (full) socket->forceClose();
                    else listen(socket, inbox);
                });
                ready.set_value(true); announced = true;
                std::optional<Time::time_point> shutdown;
                while (!mail->quit) {
                    auto now = Time::now();
                    if (mail->leaving && !shutdown) {
                        shutdown = now;
                        server.stop();
                        for (auto& room : rooms) { finish(*room, 0); room->closing = now; }
                    }
                    std::deque<Socket> accepted;
                    std::deque<Packet> packets;
                    {
                        std::lock_guard lock(mail->mutex);
                        accepted.swap(mail->accepted); packets.swap(mail->packets); mail->bytes = 0;
                    }
                    for (auto& socket : accepted) {
                        if (clients.size() >= 128 || shutdown) { socket->close(); continue; }
                        auto client = std::make_shared<Client>(); client->socket = socket; clients.push_back(client);
                    }
                    for (auto& packet : packets) {
                        auto found = std::find_if(clients.begin(), clients.end(), [&](const auto& client) { return client->socket == packet.socket; });
                        if (found == clients.end() || !packet.socket->isOpen()) continue;
                        auto client = *found;
                        try {
                            auto message = parse(packet.text);
                            auto type = message.at("type").get<std::string>();
                            client->last = now;
                            if (client->seat < 0) {
                                require(!shutdown && type == "hello", "Expected LAN room identification");
                                auto id = message.at("room").get<std::string>();
                                client->name = message.at("name").get<std::string>();
                                int duration = message.at("milliseconds").get<int>();
                                require(nameValid(client->name) && (id.empty() || nameValid(id)), "Invalid LAN username or room ID");
                                require(duration > 0 && duration <= 60000, "Invalid LAN half-turn duration");
                                Room* target = nullptr;
                                // arrival order determines the creator. empty IDs only select public vacancies.
                                for (auto& room : rooms) {
                                    if (id.empty() ? (room->publicRoom && !room->closing && room->players[0] && !room->players[1])
                                                   : (!room->publicRoom && room->label == id)) { target = room.get(); break; }
                                }
                                if (!target) {
                                    require(rooms.size() < 64, "LAN lobby reached its room limit");
                                    auto room = std::make_unique<Room>();
                                    room->publicRoom = id.empty();
                                    room->label = id.empty() ? std::to_string(++number) : id;
                                    room->key = (id.empty() ? "public:" : "private:") + room->label;
                                    room->milliseconds = duration;
                                    target = room.get(); rooms.push_back(std::move(room));
                                }
                                require(!target->closing && !target->players[1], "Room is full");
                                client->seat = target->players[0] ? 1 : 0;
                                client->room = target->key;
                                target->players[client->seat] = client;
                                target->names[client->seat] = client->name;
                                ++target->revision;
                                send(client->socket, {{"type", "welcome"}, {"room", target->label}, {"player", client->seat}});
                                continue;
                            }
                            auto foundRoom = std::find_if(rooms.begin(), rooms.end(), [&](const auto& room) { return room->key == client->room; });
                            require(foundRoom != rooms.end(), "LAN room closed");
                            auto& room = **foundRoom;
                            if (type == "ping") { send(client->socket, {{"type", "pong"}}); continue; }
                            if (type == "recorded") {
                                require(!room.match.snapshot().running && message.at("game").get<std::uint64_t>() == room.game &&
                                        client->offset == room.replay.size(), "Unexpected recording acknowledgement");
                                client->recorded = room.game;
                                continue;
                            }
                            if (type == "leave") {
                                finish(room, client->seat); client->leaving = true;
                                if (client->seat == 0) room.closing = now;
                            } else if (type == "stop") finish(room, client->seat);
                            else if (type == "ready") {
                                if (!room.match.snapshot().running && !room.closing && !client->leaving) { client->ready = true; ++room.revision; }
                            } else if (type == "move" || type == "cancel") {
                                require(message.at("id").is_number_unsigned(), "Invalid LAN input number");
                                auto id = message["id"].get<std::uint64_t>();
                                require(id > client->ack, "LAN input number moved backwards");
                                client->ack = id; ++room.revision;
                                if (message.at("game").get<std::uint64_t>() != room.game || client->game != room.game || client->leaving || room.closing) continue;
                                if (message.contains("tick") && message["tick"].get<std::uint64_t>() != (*room.match.snapshot().views)[client->seat].tick) continue;
                                if (type == "move") room.match.enqueue(client->seat, decode(message.at("action")));
                                else room.match.cancel(client->seat, message.at("all").get<bool>());
                            } else throw std::runtime_error("Unexpected LAN client command");
                        } catch (const std::exception& error) {
                            try { send(client->socket, {{"type", "error"}, {"message", error.what()}}); } catch (...) {}
                            client->socket->close();
                        }
                    }
                    for (auto& owner : rooms) {
                        auto& room = *owner;
                        try {
                        for (int seat = 0; seat < 2; ++seat) {
                            auto client = room.players[seat];
                            if (client && (!client->socket->isOpen() || now - client->last > 20s)) {
                                client->socket->close(); detach(room, seat);
                            }
                        }
                        auto local = room.match.snapshot();
                        if (room.game && local.state == MatchState::Finished && !room.finished) {
                            room.finished = true;
                            for (auto& client : room.players) if (client) client->ready = false;
                            ++room.revision;
                        }
                        if (!room.closing && room.players[0] && room.players[1] && room.players[0]->ready && room.players[1]->ready &&
                            !local.running && std::all_of(room.players.begin(), room.players.end(), [&](const auto& client) {
                                return !client->leaving && (!room.game || client->game != room.game || client->recorded == room.game);
                            })) {
                            require(room.match.start({}, std::random_device{}(), room.milliseconds, {}, true), "LAN map initialization failed");
                            ++room.game; ++room.revision; room.finished = false; room.replay.clear();
                            for (auto& client : room.players) { client->game = room.game; client->ready = false; client->offset = 0; }
                            local = room.match.snapshot();
                        }
                        if (local.completed && room.replay.empty()) {
                            room.replay = encodeReplay(*local.completed);
                            require(room.replay.size() <= recordingLimit, "LAN recording exceeds transfer limit");
                        }
                        const auto& names = room.names;
                        for (int seat = 0; seat < 2; ++seat) {
                            auto client = room.players[seat];
                            if (!client) continue;
                            try {
                                if (client->revision != room.revision || client->views != local.views) {
                                    if (!client->game || client->game != room.game) {
                                        send(client->socket, {{"type", "waiting"}, {"ready", client->ready}, {"names", names}});
                                    } else {
                                        auto view = (*local.views)[seat];
                                        auto outcome = view.result; view.result = Phases::Ongoing;
                                        std::ostringstream observation;
                                        require(Protocol::writeObservation(observation, view), "Cannot encode LAN observation");
                                        Json queue = Json::array();
                                        for (auto& action : local.queued[seat]) queue.push_back(encode(action));
                                        send(client->socket, {{"type", "state"}, {"game", room.game}, {"rows", view.rows}, {"cols", view.cols},
                                             {"view", observation.str()}, {"result", static_cast<int>(outcome)}, {"ready", client->ready},
                                             {"names", names}, {"queued", queue}, {"ack", client->ack}});
                                    }
                                    client->views = local.views; client->revision = room.revision;
                                }
                                // completed replays reveal the full board, so transfer starts only after game over.
                                if (client->game == room.game && !room.replay.empty() && client->offset < room.replay.size() && client->socket->bufferedAmount() < 65536) {
                                    auto end = std::min(room.replay.size(), client->offset + 8192);
                                    std::string hex;
                                    for (auto index = client->offset; index < end; ++index) {
                                        auto byte = static_cast<unsigned char>(room.replay[index]);
                                        hex += "0123456789abcdef"[byte >> 4]; hex += "0123456789abcdef"[byte & 15];
                                    }
                                    send(client->socket, {{"type", "replay"}, {"game", room.game}, {"offset", client->offset},
                                                         {"data", hex}, {"last", end == room.replay.size()}});
                                    client->offset = end;
                                }
                                bool delivered = !client->game || client->recorded == room.game;
                                if (room.closing && (delivered || now - *room.closing > 3s)) {
                                    send(client->socket, {{"type", "closed"}, {"message", "Room closed: its host left"}});
                                    client->socket->close(); room.players[seat].reset();
                                } else if (client->leaving && delivered) {
                                    send(client->socket, {{"type", "closed"}, {"message", "Left LAN room"}});
                                    client->socket->close(); room.players[seat].reset(); ++room.revision;
                                }
                            } catch (...) { client->socket->forceClose(); detach(room, seat); }
                        }
                        } catch (const std::exception& error) {
                            // failure in one match must not close the listener or another room.
                            room.closing = now;
                            room.match.stop();
                            for (auto& client : room.players) if (client) {
                                try { send(client->socket, {{"type", "error"}, {"message", error.what()}}); } catch (...) {}
                                client->socket->close(); client.reset();
                            }
                        }
                    }
                    std::erase_if(rooms, [](const auto& room) { return room->closing && !room->players[0] && !room->players[1]; });
                    std::erase_if(clients, [&](const auto& client) {
                        if (client->seat < 0 && now - client->last > 5s) client->socket->close();
                        return client->socket->isClosed();
                    });
                    if (shutdown && (rooms.empty() || now - *shutdown > 3s)) break;
                    std::this_thread::sleep_for(10ms);
                }
                server.stop();
            } catch (...) {
                if (!announced) ready.set_value(false);
            }
            mail->quit = true;
            for (auto& client : clients) client->socket->close();
            { std::lock_guard lock(mail->mutex); mail->packets.clear(); mail->accepted.clear(); }
        }
    };

    struct LanMatch::Session {
        mutable std::mutex mutex;
        MatchSnapshot published;
        std::shared_ptr<Mail> mail = std::make_shared<Mail>();
        std::thread worker;
        std::atomic<bool> open{false};
        LanConfig config;
        std::string unsaved;
        std::uint64_t game = 0, sequence = 0;
        std::deque<Json> pending;

        MatchSnapshot snapshot() const { std::lock_guard lock(mutex); return published; }
        void publish(MatchSnapshot value, std::uint64_t next = 0, std::uint64_t ack = 0) {
            std::lock_guard lock(mutex);
            game = next;
            std::erase_if(pending, [&](const Json& item) { return item["id"].get<std::uint64_t>() <= ack || item["game"].get<std::uint64_t>() != game; });
            if (value.running && value.player >= 0) {
                auto& queue = value.queued[value.player];
                // overlay unacknowledged edits, so receiving an older queue cannot erase fresh human input.
                for (auto& item : pending) {
                    if (item["type"] == "move") queue.push_back(decode(item["action"]));
                    else if (item["all"].get<bool>()) queue.clear();
                    else if (!queue.empty()) queue.pop_back();
                }
            }
            published = std::move(value);
        }
        bool command(Json value) {
            std::lock_guard stateLock(mutex);
            std::lock_guard inboxLock(mail->mutex);
            if (mail->commands.size() >= Capacity || pending.size() >= Capacity) return false;
            value["id"] = ++sequence; value["game"] = game;
            if (value["type"] == "move" || value["type"] == "cancel") {
                if (published.player < 0) return false;
                pending.push_back(value);
                auto& queue = published.queued[published.player];
                if (value["type"] == "move") queue.push_back(decode(value["action"]));
                else if (value["all"].get<bool>()) queue.clear();
                else if (!queue.empty()) queue.pop_back();
            }
            mail->commands.push_back(std::move(value));
            return true;
        }
        void run(Address endpoint, std::vector<std::unique_ptr<Lobby>>& lobbies) {
            MatchSnapshot state;
            state.state = MatchState::Active;
            Socket socket;
            StrategyProcess agent;
            bool welcomed = false, agentStarted = false, replied = false;
            std::uint64_t current = 0, saved = 0, requested = 0, ack = 0;
            std::string recording, roomLabel;
            auto directory = config.directory, nextDirectory = config.directory;
            std::optional<Time::time_point> leaving;
            auto last = Time::now(), heartbeat = last;
            try {
                auto proxy = httpProxy(config.proxy);
                socket = connect(endpoint, mail, 2s, proxy);
                // proxy failures must not bypass the configured route or open a different local lobby.
                if (!socket && !proxy && endpoint.listen && !mail->leaving) {
                    auto lobby = std::make_unique<Lobby>(endpoint);
                    if (lobby->opened) lobbies.push_back(std::move(lobby));
                }
                // another local instance may have reserved the port just before it starts listening.
                auto retryUntil = Time::now() + 10s;
                while (!socket && !mail->leaving && Time::now() < retryUntil) {
                    socket = connect(endpoint, mail, 2s, proxy);
                    if (!socket) std::this_thread::sleep_for(100ms);
                }
                require(socket != nullptr, proxy ? "Cannot connect to LAN IP through the proxy" : "Cannot connect to LAN IP or open a local listener");
                send(socket, {{"type", "hello"}, {"name", config.username}, {"room", config.room}, {"milliseconds", config.milliseconds}});
                last = heartbeat = Time::now();
                while (!mail->quit) {
                    auto now = Time::now();
                    if (mail->leaving && !leaving) { leaving = now; send(socket, {{"type", "leave"}}); }
                    if (leaving && now - *leaving > 4s) break;
                    std::deque<Packet> packets;
                    std::deque<Json> commands;
                    {
                        std::lock_guard lock(mail->mutex);
                        packets.swap(mail->packets); commands.swap(mail->commands); mail->bytes = 0;
                    }
                    for (auto& packet : packets) {
                        if (packet.socket != socket) continue;
                        auto message = parse(packet.text);
                        auto type = message.at("type").get<std::string>();
                        last = now;
                        if (type == "error") throw std::runtime_error(message.at("message").get<std::string>());
                        if (type == "closed") { state.status = message.at("message").get<std::string>(); mail->quit = true; break; }
                        if (type == "pong") continue;
                        if (type == "welcome") {
                            require(!welcomed, "Repeated LAN welcome");
                            state.player = message.at("player").get<int>();
                            require(state.player == 0 || state.player == 1, "Invalid LAN seat");
                            state.host = 0;
                            roomLabel = message.at("room").get<std::string>();
                            require(nameValid(roomLabel), "Invalid LAN room label");
                            welcomed = true;
                        } else if (type == "waiting") {
                            require(welcomed, "LAN waiting state before welcome");
                            state.names = message.at("names").get<std::array<std::string, 2>>();
                            state.running = false;
                            state.state = message.at("ready").get<bool>() ? MatchState::Active : MatchState::Finished;
                            state.status = "Room " + roomLabel + " - Waiting for peer";
                        } else if (type == "state") {
                            require(welcomed, "LAN state before welcome");
                            auto next = message.at("game").get<std::uint64_t>();
                            require(next > 0 && next >= current, "Invalid LAN game number");
                            int rows = message.at("rows").get<int>(), cols = message.at("cols").get<int>();
                            std::istringstream input(message.at("view").get<std::string>());
                            auto view = Protocol::readObservation(input, {state.player, rows, cols}); input >> std::ws;
                            require(view.has_value() && input.eof(), "Invalid LAN observation");
                            int result = message.at("result").get<int>(); require(result >= 0 && result <= 3, "Invalid LAN result");
                            view->result = static_cast<Phases>(result);
                            if (next != current) {
                                agent.stop(); agentStarted = false; recording.clear(); directory = nextDirectory; state.error.clear();
                            } else if (current) require(view->tick >= (*state.views)[state.player].tick, "LAN tick moved backwards");
                            current = next;
                            auto views = std::make_shared<std::array<Observation, 3>>(); (*views)[state.player] = *view; state.views = views;
                            state.names = message.at("names").get<std::array<std::string, 2>>();
                            require(nameValid(state.names[0]) && nameValid(state.names[1]), "Invalid LAN player names");
                            state.running = result == 0;
                            state.state = state.running || message.at("ready").get<bool>() ? MatchState::Active : MatchState::Finished;
                            state.status = "Room " + roomLabel + " - " + (result == 0 ? "LAN match" : result == 1 ? "Red wins" : result == 2 ? "Blue wins" : "Draw");
                            const auto& queue = message.at("queued"); require(queue.is_array() && queue.size() <= Capacity, "Invalid LAN queue");
                            state.queued[state.player].clear();
                            for (auto& action : queue) state.queued[state.player].push_back(decode(action));
                            ack = message.at("ack").get<std::uint64_t>();
                        } else if (type == "replay") {
                            require(current && !state.running && message.at("game").get<std::uint64_t>() == current && saved != current, "Unexpected LAN recording");
                            auto offset = message.at("offset").get<std::size_t>();
                            auto hex = message.at("data").get<std::string>();
                            require(offset == recording.size() && hex.size() <= 16384 && hex.size() % 2 == 0 &&
                                    offset + hex.size() / 2 <= recordingLimit, "Invalid LAN recording chunk");
                            for (std::size_t index = 0; index < hex.size(); index += 2) {
                                auto digit = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
                                int a = digit(hex[index]), b = digit(hex[index + 1]); require(a >= 0 && b >= 0, "Invalid LAN recording bytes");
                                recording += static_cast<char>((a << 4) | b);
                            }
                            if (message.at("last").get<bool>()) {
                                if (!directory.empty()) {
                                    Replay replay;
                                    std::string error;
                                    require(replay.loadBytes(recording, error), "Invalid LAN recording");
                                    if (auto path = saveReplayBytes(recording, directory, error)) state.status += " - Saved " + path->filename().string();
                                    else {
                                        state.error = error;
                                        std::lock_guard lock(mutex);
                                        unsaved = recording;
                                    }
                                }
                                saved = current;
                                if (state.error.empty()) recording.clear();
                                send(socket, {{"type", "recorded"}, {"game", current}});
                            }
                        } else throw std::runtime_error("Unexpected LAN server packet");
                        publish(state, current, ack);
                    }
                    for (auto& command : commands) {
                        if (leaving || mail->quit) break;
                        if (command["type"] == "ready") {
                            auto path = command.at("directory").get<std::string>(); nextDirectory = std::u8string(path.begin(), path.end());
                            command.erase("directory");
                            if (current && saved != current) { state.error = "Recording is still arriving; retry Reset shortly"; publish(state, current, ack); continue; }
                            if (!recording.empty()) {
                                std::string error;
                                if (nextDirectory.empty() || !saveReplayBytes(recording, nextDirectory, error)) {
                                    state.error = error.empty() ? "Set RD to save the pending recording before Reset" : error;
                                    publish(state, current, ack); continue;
                                }
                                recording.clear();
                                { std::lock_guard lock(mutex); unsaved.clear(); }
                            }
                            state.error.clear();
                        }
                        send(socket, command);
                    }
                    if (welcomed && state.running && !config.command.empty() && !leaving && !mail->quit) {
                        try {
                            const auto& view = (*state.views)[state.player];
                            if (!agentStarted) {
                                require(agent.start(config.command, {state.player, view.rows, view.cols}), "Cannot start LAN agent");
                                agentStarted = true; requested = view.tick; replied = false; agent.request(view);
                            } else if (requested != view.tick) { requested = view.tick; replied = false; agent.request(view); }
                            auto error = agent.error(); if (!error.empty()) throw std::runtime_error(error);
                            if (!replied) if (auto action = agent.reply(requested, now)) {
                                replied = true;
                                if (action->type == ActionType::Move) command({{"type", "move"}, {"tick", requested}, {"action", encode(*action)}});
                            }
                        } catch (const std::exception& error) {
                            agent.stop(); agentStarted = false; send(socket, {{"type", "stop"}});
                            state.error = error.what(); state.running = false; publish(state, current, ack);
                        }
                    } else if (agentStarted) { agent.stop(); agentStarted = false; }
                    if (!socket->isOpen() && !mail->quit) throw std::runtime_error("LAN lobby disconnected");
                    if (now - last > (welcomed ? 20s : 10s)) throw std::runtime_error("LAN connection timed out");
                    if (now - heartbeat > 5s && !mail->quit) { send(socket, {{"type", "ping"}}); heartbeat = now; }
                    std::this_thread::sleep_for(10ms);
                }
            } catch (const std::exception& error) { state.error = error.what(); state.status = "LAN connection ended"; }
            mail->quit = true; agent.stop(); if (socket) socket->close();
            { std::lock_guard lock(mail->mutex); mail->packets.clear(); mail->commands.clear(); }
            state.state = MatchState::Finished; state.running = false; publish(state, current, ack); open = false;
        }
    };

    LanMatch::LanMatch() : session(std::make_unique<Session>()) {}
    LanMatch::~LanMatch() { leave(); }
    bool LanMatch::start(const LanConfig& config) {
        if (opened()) return false;
        leave();
        std::string pendingError;
        if (!savePending(config.directory, pendingError)) {
            auto state = snapshot(); state.error = pendingError; session->publish(state); return false;
        }
        session = std::make_unique<Session>();
        try {
            require(nameValid(config.username), "LAN username must contain 1-64 printable ASCII characters");
            require(config.room.empty() || nameValid(config.room), "Room ID must contain at most 64 printable ASCII characters");
            require(config.milliseconds > 0 && config.milliseconds <= 60000, "LAN half-turn must be between 1 and 60000 ms");
            std::string error;
            require(config.directory.empty() || replayDirectory(config.directory, error), "Recording directory is unavailable");
            require(config.command.empty() || StrategyProcess::available(config.command), "Cannot open LAN player program");
            httpProxy(config.proxy);
            auto endpoint = address(config.address); session->config = config; session->open = true;
            MatchSnapshot pending; pending.state = MatchState::Active; pending.status = "Connecting to LAN lobby"; session->publish(pending);
            session->worker = std::thread([this, endpoint] { session->run(endpoint, lobbies); });
            return true;
        } catch (const std::exception& error) {
            session->open = false; MatchSnapshot failed; failed.error = error.what(); session->publish(failed); return false;
        }
    }
    void LanMatch::leave() {
        session->mail->leaving = true;
        if (session->worker.joinable()) session->worker.join();
        session->open = false;
    }
    bool LanMatch::opened() const { return session->open; }
    MatchSnapshot LanMatch::snapshot() const { return session->snapshot(); }
    bool LanMatch::savePending(const std::filesystem::path& directory, std::string& error) {
        std::lock_guard lock(session->mutex);
        if (session->unsaved.empty()) return true;
        if (directory.empty()) { error = "Set RD to save the pending LAN recording"; return false; }
        if (!saveReplayBytes(session->unsaved, directory, error)) return false;
        session->unsaved.clear();
        return true;
    }
    void LanMatch::stop() { if (opened()) session->command({{"type", "stop"}}); }
    void LanMatch::restart(const std::filesystem::path& directory) {
        auto path = directory.u8string();
        if (opened()) session->command({{"type", "ready"}, {"directory", std::string(path.begin(), path.end())}});
    }
    bool LanMatch::enqueue(int player, const Action& action) {
        auto state = snapshot();
        if (!opened() || player < 0 || player > 1 || player != state.player || !state.running || !session->config.command.empty()) return false;
        const auto& view = (*state.views)[player];
        int row = action.row, col = action.col;
        if (action.type != ActionType::Move || row < 0 || col < 0 || row >= view.rows || col >= view.cols) return false;
        switch (action.direction) {
            case Direction::Up: --row; break;
            case Direction::Down: ++row; break;
            case Direction::Left: --col; break;
            case Direction::Right: ++col; break;
            default: return false;
        }
        if (row < 0 || col < 0 || row >= view.rows || col >= view.cols || state.queued[player].size() >= Capacity) return false;
        return session->command({{"type", "move"}, {"action", encode(action)}});
    }
    std::optional<Action> LanMatch::cancel(int player, bool all) {
        auto state = snapshot();
        if (!opened() || player < 0 || player > 1 || player != state.player || !state.running || state.queued[player].empty()) return std::nullopt;
        auto removed = all ? state.queued[player].front() : state.queued[player].back();
        if (!session->command({{"type", "cancel"}, {"all", all}})) return std::nullopt;
        return removed;
    }
}
