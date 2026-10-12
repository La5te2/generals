// authoritative room service. the lobby owns admission and pairing. each paired room owns its worker and engine.
#include "server.hpp"
#include "interface/local.hpp"
#include "wire.hpp"
#include <rtc/websocketserver.hpp>
#include <atomic>
#include <random>

namespace NEBULA {
    namespace {
        using namespace Wire;
        using namespace std::chrono_literals;
        struct Player {
            // the connection identifies a participant. its room-array index determines the seat.
            // names are display labels and may be identical, including within one room.
            std::shared_ptr<Connection> connection;
            std::string name, room;
            bool joined = false;
        };
        struct Room {
            std::string id, key; // key is a private Room ID. empty denotes public matchmaking.
            std::array<Player, 2> players;
            std::atomic<bool> quit{false}, done{false};
            std::thread worker;
            // transfer two matched connections from the lobby into one independently running room.
            Room(std::string identity, Player red, Player blue)
                : id(std::move(identity)), key(red.room), players{std::move(red), std::move(blue)} {
                worker = std::thread([this] { run(); });
            }
            // may wait for game cleanup. the lobby normally destroys rooms only after done is set.
            ~Room() { quit = true; if (worker.joinable()) worker.join(); }
            // wait for both ready messages, route seat-specific input and stream each player's observation.
            // departure forfeits that seat. completion sends a recording before closing each surviving connection.
            // all exits set done so the lobby can reclaim this room and its private Room ID.
            void run() noexcept {
                LocalMatch match;
                try {
                    require(match.start({}, std::random_device{}(), halfTurnMilliseconds, {}, true), "Map initialization failed");
                    match.pause();
                    std::array<std::string, 2> names{players[0].name, players[1].name};
                    std::array<bool, 2> ready{}, closed{}, recorded{}, finalSent{};
                    std::array<std::uint64_t, 2> ack{}, sentAck{};
                    std::array<std::size_t, 2> offsets{};
                    std::array<std::shared_ptr<const std::array<Observation, 3>>, 2> sentViews;
                    std::string replay;
                    std::optional<Time::time_point> finished;
                    auto began = Time::now();
                    bool started = false;
                    auto end = [&](int loser) {
                        if (!finished) { match.stop(loser); finished = Time::now(); }
                    };
                    for (int seat = 0; seat < 2; ++seat) {
                        try { players[seat].connection->send({{"type", "paired"}, {"room", id}, {"player", seat}, {"names", names}}); }
                        catch (...) { closed[seat] = true; end(seat); }
                    }
                    while (!quit) {
                        auto now = Time::now();
                        for (int seat = 0; seat < 2; ++seat) {
                            if (closed[seat]) continue;
                            auto& peer = *players[seat].connection;
                            try {
                                for (auto& text : peer.take()) {
                                    auto message = parse(text);
                                    if (peer.control(message)) continue;
                                    auto type = message.at("type").get<std::string>();
                                    if (type == "leave") end(seat);
                                    else if (type == "ready") {
                                        require(!ready[seat] && !started, "Unexpected ready packet");
                                        ready[seat] = true;
                                    } else if (type == "recorded") {
                                        require(finished && !replay.empty() && offsets[seat] == replay.size(), "Unexpected recording acknowledgement");
                                        recorded[seat] = true;
                                    } else if (type == "move" || type == "cancel") {
                                        require(ready[seat], "Input before ready");
                                        require(message.at("id").is_number_unsigned(), "Invalid input sequence");
                                        auto sequence = message["id"].get<std::uint64_t>();
                                        require(sequence > ack[seat], "Input sequence moved backwards");
                                        ack[seat] = sequence;
                                        auto local = match.snapshot();
                                        if (finished || !local.running()) continue;
                                        if (message.contains("tick")) {
                                            require(message["tick"].is_number_unsigned(), "Invalid observation tick");
                                            if (message["tick"].get<std::uint64_t>() != (*local.views)[seat].tick) continue;
                                        }
                                        if (type == "move") match.enqueue(seat, decode(message.at("action")));
                                        else match.cancel(seat, message.at("all").get<bool>());
                                    } else throw std::runtime_error("Unexpected player packet");
                                }
                                require(peer.socket->isOpen(), "Player disconnected");
                                peer.pulse();
                                if (!ready[seat] && now - began > 10s) throw std::runtime_error("Player did not become ready");
                            } catch (const std::exception& error) { peer.reject(error.what()); closed[seat] = true; end(seat); }
                        }
                        if (ready[0] && ready[1] && !started && !finished) { match.resume(); started = true; }
                        auto local = match.snapshot();
                        if (local.state == MatchState::Finished && !finished) finished = now;
                        if (finished && replay.empty()) {
                            require(local.completed != nullptr, "Completed recording is unavailable");
                            auto record = *local.completed; record.names = names;
                            replay = encodeReplay(record);
                            require(!replay.empty() && replay.size() <= recordingLimit, "Recording exceeds transfer limit");
                        }
                        for (int seat = 0; seat < 2; ++seat) {
                            if (closed[seat] || (!started && !finished)) continue;
                            auto& peer = *players[seat].connection;
                            try {
                                if (!finalSent[seat] && (sentViews[seat] != local.views || sentAck[seat] != ack[seat])) {
                                    auto view = (*local.views)[seat];
                                    auto result = view.result; view.result = Phases::Ongoing;
                                    std::ostringstream observation;
                                    require(Protocol::writeObservation(observation, view), "Cannot encode observation");
                                    Json queue = Json::array();
                                    for (auto& action : local.queued[seat]) queue.push_back(encode(action));
                                    peer.send({{"type", "state"}, {"rows", view.rows}, {"cols", view.cols}, {"view", observation.str()},
                                               {"result", static_cast<int>(result)}, {"queued", queue}, {"ack", ack[seat]}});
                                    sentViews[seat] = local.views; sentAck[seat] = ack[seat];
                                    finalSent[seat] = result != Phases::Ongoing;
                                }
                                if (finished && finalSent[seat] && offsets[seat] < replay.size() && peer.socket->bufferedAmount() < 128 * 1024) {
                                    auto count = std::min(chunkLimit, replay.size() - offsets[seat]);
                                    peer.send({{"type", "replay"}, {"offset", offsets[seat]},
                                               {"data", hex(std::string_view(replay).substr(offsets[seat], count))},
                                               {"last", offsets[seat] + count == replay.size()}});
                                    offsets[seat] += count;
                                }
                                if (recorded[seat]) { peer.send({{"type", "closed"}}); peer.socket->close(); closed[seat] = true; }
                            } catch (const std::exception& error) { peer.reject(error.what()); closed[seat] = true; end(seat); }
                        }
                        if (closed[0] && closed[1]) break;
                        if (finished && now - *finished > 30s) throw std::runtime_error("Recording transfer timed out");
                        std::this_thread::sleep_for(10ms);
                    }
                    if (quit) for (auto& player : players) player.connection->reject("Server is shutting down");
                } catch (const std::exception& error) {
                    for (auto& player : players) player.connection->reject(error.what());
                }
                done = true;
            }
        };
        struct Lobby {
            std::mutex mutex;
            std::deque<std::shared_ptr<Connection>> accepted;
            bool closing = false;
        };
    }

    struct RoomServer::Impl {
        std::shared_ptr<Lobby> lobby = std::make_shared<Lobby>();
        std::unique_ptr<rtc::WebSocketServer> listener;
        std::thread worker;
        std::atomic<bool> quit{false};
        std::vector<Player> waiting;
        std::vector<std::unique_ptr<Room>> rooms;
        std::uint64_t sequence = 0;

        // WebSocket callbacks only enqueue accepted connections. the lobby worker owns admission decisions.
        Impl(const std::string& address, std::uint16_t port) {
            rtc::WebSocketServer::Configuration config;
            config.bindAddress = address; config.port = port; config.maxMessageSize = packetLimit; config.connectionTimeout = 10s;
            listener = std::make_unique<rtc::WebSocketServer>(config);
            listener->onClient([mail = lobby](Socket socket) {
                auto connection = std::make_shared<Connection>(std::move(socket));
                std::lock_guard lock(mail->mutex);
                if (mail->closing || mail->accepted.size() >= 32) connection->reject("Server admission queue is full");
                else mail->accepted.push_back(std::move(connection));
            });
            worker = std::thread([this] { run(); });
        }
        // close admission before stopping the listener, then let the lobby drain its players and room workers.
        ~Impl() {
            { std::lock_guard lock(lobby->mutex); lobby->closing = true; }
            listener->stop(); quit = true;
            if (worker.joinable()) worker.join();
        }
        // validate joins and pair waiting players with equal Room IDs in arrival order. empty IDs form public pairs.
        // a live private room reserves its ID. paired sockets leave this loop and belong exclusively to their room.
        void run() noexcept {
            try {
                while (!quit) {
                    std::erase_if(rooms, [](const auto& room) { return room->done.load(); });
                    std::deque<std::shared_ptr<Connection>> accepted;
                    { std::lock_guard lock(lobby->mutex); accepted.swap(lobby->accepted); }
                    for (auto& connection : accepted) {
                        if (waiting.size() + rooms.size() * 2 >= 128) connection->reject("Server is full");
                        else waiting.push_back({std::move(connection), {}, {}, false});
                    }
                    for (auto& player : waiting) {
                        auto& peer = *player.connection;
                        try {
                            for (auto& text : peer.take()) {
                                auto message = parse(text);
                                if (peer.control(message)) continue;
                                auto type = message.at("type").get<std::string>();
                                if (type == "leave") { peer.send({{"type", "closed"}}); peer.socket->close(); break; }
                                require(type == "join" && !player.joined, "Expected one join packet");
                                require(message.at("version") == version, "Unsupported custom-server protocol version");
                                player.name = message.at("name").get<std::string>(); player.room = message.at("room").get<std::string>();
                                require(nameValid(player.name) && (player.room.empty() || nameValid(player.room)), "Invalid username or Room ID");
                                require(std::none_of(rooms.begin(), rooms.end(), [&](const auto& room) {
                                    return !player.room.empty() && room->key == player.room;
                                }), "Room is full");
                                player.joined = true;
                                peer.send({{"type", "waiting"}});
                            }
                            if (!player.joined && Time::now() - peer.arrived > 10s) throw std::runtime_error("Join timed out");
                            if (peer.socket->isOpen()) peer.pulse();
                        } catch (const std::exception& error) { peer.reject(error.what()); }
                    }
                    // onClient may run before the accepted socket's handshake has completed.
                    std::erase_if(waiting, [](const Player& player) { return player.connection->socket->readyState() >= rtc::WebSocket::State::Closing; });
                    for (std::size_t first = 0; first < waiting.size();) {
                        if (!waiting[first].joined) { ++first; continue; }
                        auto second = std::find_if(waiting.begin() + first + 1, waiting.end(), [&](const Player& other) {
                            return other.joined && other.room == waiting[first].room;
                        });
                        if (second == waiting.end()) { ++first; continue; }
                        auto room = std::make_unique<Room>(std::to_string(++sequence), std::move(waiting[first]), std::move(*second));
                        waiting.erase(second); waiting.erase(waiting.begin() + first);
                        // close remaining contenders for this private ID in the same lobby iteration.
                        if (!room->key.empty()) {
                            for (auto& other : waiting) if (other.joined && other.room == room->key) other.connection->reject("Room is full");
                            std::erase_if(waiting, [](const Player& player) { return player.connection->socket->readyState() >= rtc::WebSocket::State::Closing; });
                        }
                        rooms.push_back(std::move(room));
                    }
                    std::this_thread::sleep_for(10ms);
                }
            } catch (const std::exception& error) {
                for (auto& player : waiting) player.connection->reject(error.what());
            }
            for (auto& player : waiting) player.connection->reject("Server is shutting down");
            for (auto& room : rooms) room->quit = true;
            rooms.clear(); waiting.clear();
        }
    };
    RoomServer::RoomServer(const std::string& address, std::uint16_t port) : impl(std::make_unique<Impl>(address, port)) {}
    RoomServer::~RoomServer() = default;
    std::uint16_t RoomServer::port() const { return impl->listener->port(); }
}
