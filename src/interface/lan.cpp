// Discover peers on a selected LAN and exchange game data through one direct WebSocket connection.
// The accepting participant runs the engine. Each window owns its match, network worker and recording.
#include "lan.hpp"
#include "discovery.hpp"
#include "local.hpp"
#include "proxy.hpp"
#include <rtc/websocket.hpp>
#include <rtc/websocketserver.hpp>
#include <nlohmann/json.hpp>
#include <atomic>
#include <deque>
#include <random>

namespace NEBULA {
    namespace {
        using Json = nlohmann::json;
        using Socket = std::shared_ptr<rtc::WebSocket>;
        using Time = std::chrono::steady_clock;
        using namespace std::chrono_literals;
        constexpr std::size_t packetLimit = 256 * 1024, recordingLimit = 16 * 1024 * 1024;
        constexpr int halfTurnMilliseconds = 500;

        void require(bool value, const char* error) { if (!value) throw std::runtime_error(error); }
        bool nameValid(const std::string& value) {
            return !value.empty() && value.size() <= 64 &&
                std::all_of(value.begin(), value.end(), [](unsigned char c) { return c >= 32 && c <= 126; });
        }
        std::string identity() {
            std::random_device random;
            std::string value;
            for (int i = 0; i < 32; ++i) value += "0123456789abcdef"[random() & 15];
            return value;
        }
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
                    overflow = !text || text->size() > packetLimit || mail->packets.size() >= 128 ||
                               mail->bytes + (text ? text->size() : 0) > 2 * 1024 * 1024;
                    if (!overflow) { mail->bytes += text->size(); mail->packets.push_back({source, std::move(*text)}); }
                }
                if (overflow) source->forceClose();
            });
        }
        void send(const Socket& socket, const Json& value) {
            require(socket && socket->isOpen(), "LAN peer disconnected");
            require(socket->bufferedAmount() < 2 * 1024 * 1024, "LAN peer is not receiving updates");
            socket->send(value.dump());
        }
        Json parse(const std::string& text) {
            auto data = Json::parse(text, [](int depth, Json::parse_event_t, Json&) {
                require(depth <= 12, "LAN packet nesting is too deep"); return true;
            });
            require(data.is_object(), "Expected LAN packet object");
            return data;
        }
        std::string textField(const Json& message, const char* name) {
            auto field = message.find(name);
            return field != message.end() && field->is_string() ? field->get<std::string>() : "";
        }
        std::string hex(std::string_view bytes) {
            std::string result;
            result.reserve(bytes.size() * 2);
            for (unsigned char byte : bytes) {
                result += "0123456789abcdef"[byte >> 4]; result += "0123456789abcdef"[byte & 15];
            }
            return result;
        }
        void appendBytes(std::string& bytes, const Json& message) {
            auto offset = message.at("offset").get<std::size_t>();
            auto text = message.at("data").get<std::string>();
            require(offset == bytes.size() && text.size() <= 16384 && text.size() % 2 == 0 &&
                    offset + text.size() / 2 <= recordingLimit, "Invalid LAN recording chunk");
            for (std::size_t i = 0; i < text.size(); i += 2) {
                auto digit = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
                int a = digit(text[i]), b = digit(text[i + 1]);
                require(a >= 0 && b >= 0, "Invalid LAN recording bytes");
                bytes += static_cast<char>((a << 4) | b);
            }
        }
    }

    struct LanMatch::Session {
        mutable std::mutex mutex;
        MatchSnapshot published;
        std::shared_ptr<Mail> mail = std::make_shared<Mail>();
        std::thread worker;
        std::atomic<bool> open{false};
        LanConfig config;
        std::string unsaved;
        std::uint64_t sequence = 0;
        std::deque<Json> pending;

        ~Session() { mail->leaving = true; if (worker.joinable()) worker.join(); }
        MatchSnapshot snapshot() const { std::lock_guard lock(mutex); return published; }
        void publish(MatchSnapshot value, std::uint64_t ack = 0) {
            std::lock_guard lock(mutex);
            std::erase_if(pending, [&](const Json& item) { return item["id"].get<std::uint64_t>() <= ack; });
            if (value.running() && value.player >= 0) {
                auto& queue = value.queued[value.player];
                // Preserve input submitted after the host prepared this snapshot.
                for (auto& item : pending) {
                    if (item["type"] == "move") queue.push_back(decode(item["action"]));
                    else if (item["all"].get<bool>()) queue.clear();
                    else if (!queue.empty()) queue.pop_back();
                }
            } else pending.clear();
            published = std::move(value);
        }
        bool command(Json value) {
            std::lock_guard stateLock(mutex);
            std::lock_guard inboxLock(mail->mutex);
            if (mail->leaving || !published.running() || mail->commands.size() >= Capacity || pending.size() >= Capacity) return false;
            value["id"] = ++sequence;
            pending.push_back(value);
            auto& queue = published.queued[published.player];
            if (value["type"] == "move") queue.push_back(decode(value["action"]));
            else if (value["all"].get<bool>()) queue.clear();
            else if (!queue.empty()) queue.pop_back();
            mail->commands.push_back(std::move(value));
            return true;
        }
        void persist(const std::string& bytes, MatchSnapshot& state) {
            if (config.directory.empty()) return;
            std::string error;
            if (!saveReplayBytes(bytes, config.directory, error)) {
                std::lock_guard lock(mutex);
                unsaved = bytes;
                state.error = error;
            }
        }

        void run() {
            MatchSnapshot state;
            state.state = MatchState::Waiting;
            state.status = "Searching on LAN";
            state.names = {config.username, ""};
            Socket peer;
            LocalMatch match;
            StrategyProcess agent;
            std::unique_ptr<rtc::WebSocketServer> server;
            std::string replay;
            bool paired = false, hosting = false, saved = false, received = false;
            std::uint64_t localAck = 0, remoteAck = 0;
            std::shared_ptr<const std::array<Observation, 3>> sentViews;
            struct Incoming { Socket socket; Time::time_point arrived; };
            std::vector<Incoming> incoming;
            try {
                const auto id = identity();
                auto proxy = httpProxy(config.proxy);
                rtc::WebSocketServer::Configuration options;
                options.port = 0; options.bindAddress = config.address.empty() ? "0.0.0.0" : config.address;
                options.connectionTimeout = 5s; options.maxMessageSize = packetLimit;
                server = std::make_unique<rtc::WebSocketServer>(options);
                server->onClient([inbox = mail](Socket socket) {
                    listen(socket, inbox);
                    bool full;
                    {
                        std::lock_guard lock(inbox->mutex);
                        full = inbox->quit || inbox->accepted.size() >= 8;
                        if (!full) inbox->accepted.push_back(socket);
                    }
                    if (full) socket->forceClose();
                });
                Discovery discovery(config.address, id, config.room, server->port());
                std::string target, host;
                bool helloSent = false, agentStarted = false, replied = false;
                std::uint64_t requested = 0, sentAck = 0;
                std::size_t offset = 0;
                auto began = Time::now(), attempt = began, last = began, heartbeat = began;
                auto retry = began;
                std::optional<Time::time_point> finished, leaving;
                publish(state);

                auto finish = [&](int loser) {
                    if (hosting && paired) match.stop(loser);
                    if (!finished) finished = Time::now();
                };
                auto apply = [&](const Json& command, int seat, std::uint64_t& ack) {
                    require(command.at("id").is_number_unsigned(), "Invalid LAN input number");
                    auto next = command["id"].get<std::uint64_t>();
                    require(next > ack, "LAN input number moved backwards");
                    ack = next;
                    auto local = match.snapshot();
                    if (finished || !local.running()) return;
                    if (command.contains("tick") && command["tick"].get<std::uint64_t>() != (*local.views)[seat].tick) return;
                    if (command["type"] == "move") match.enqueue(seat, decode(command.at("action")));
                    else match.cancel(seat, command.at("all").get<bool>());
                };

                while (!mail->quit) {
                    auto now = Time::now();
                    discovery.poll(!paired && !peer && !mail->leaving, paired ? host : "");
                    if (mail->leaving && !leaving) {
                        leaving = now;
                        if (!paired) break;
                        if (hosting) finish(0);
                        else send(peer, {{"type", "leave"}});
                    }
                    if (leaving && now - *leaving > 5s) break;
                    std::deque<Socket> accepted;
                    std::deque<Packet> packets;
                    std::deque<Json> commands;
                    {
                        std::lock_guard lock(mail->mutex);
                        accepted.swap(mail->accepted); packets.swap(mail->packets); commands.swap(mail->commands); mail->bytes = 0;
                    }
                    for (auto& socket : accepted) {
                        if (paired || peer || leaving || incoming.size() >= 8) {
                            try { send(socket, {{"type", "busy"}}); } catch (...) {}
                            socket->close();
                        } else incoming.push_back({socket, now});
                    }
                    for (auto& packet : packets) {
                        auto candidate = std::find_if(incoming.begin(), incoming.end(), [&](const Incoming& item) { return item.socket == packet.socket; });
                        if (packet.socket != peer && candidate == incoming.end()) continue;
                        Json message;
                        try { message = parse(packet.text); }
                        catch (...) { if (packet.socket == peer) throw; packet.socket->close(); continue; }
                        auto type = textField(message, "type");
                        if (candidate != incoming.end()) {
                            auto remoteId = textField(message, "id");
                            auto name = textField(message, "name");
                            bool known = std::any_of(discovery.peers().begin(), discovery.peers().end(), [&](const Peer& remote) {
                                return remote.id == remoteId && remote.room == config.room;
                            });
                            bool valid = type == "hello" && known && textField(message, "target") == id &&
                                textField(message, "room") == config.room && remoteId < id && nameValid(name);
                            // A private ID names one room, including while its first pair is negotiating.
                            if (!config.room.empty()) {
                                valid &= std::none_of(discovery.peers().begin(), discovery.peers().end(), [&](const Peer& remote) {
                                    return remote.id != remoteId && (!remote.available || !remote.host.empty());
                                });
                            }
                            if (!valid || peer || paired || leaving) {
                                try { send(packet.socket, {{"type", "busy"}}); } catch (...) {}
                                packet.socket->close();
                                continue;
                            }
                            peer = packet.socket; target = std::move(remoteId);
                            hosting = true; host = id; attempt = last = now;
                            state.names = {config.username, std::move(name)};
                            send(peer, {{"type", "offer"}, {"id", id}, {"target", target}, {"room", config.room}, {"names", state.names}});
                            incoming.erase(candidate);
                            continue;
                        }
                        last = now;
                        if (!paired) {
                            if (type == "busy") { peer->close(); peer.reset(); hosting = helloSent = false; retry = now + 500ms; continue; }
                            if (!hosting && type == "offer") {
                                require(message.at("id") == target && message.at("target") == id && message.at("room") == config.room, "Invalid LAN pairing offer");
                                state.names = message.at("names").get<std::array<std::string, 2>>();
                                require(nameValid(state.names[0]) && state.names[1] == config.username, "Invalid LAN player names");
                                host = target;
                                send(peer, {{"type", "accept"}, {"id", id}, {"target", target}});
                            } else if (hosting && type == "accept") {
                                require(message.at("id") == target && message.at("target") == id, "Invalid LAN pairing acceptance");
                                require(match.start({}, std::random_device{}(), halfTurnMilliseconds, {}, true), "LAN map initialization failed");
                                paired = true; state.player = state.host = 0;
                                send(peer, {{"type", "paired"}});
                            } else if (!hosting && type == "paired" && host == target) {
                                paired = true; state.player = 1; state.host = 0;
                            } else throw std::runtime_error("Unexpected LAN pairing packet");
                            if (paired) {
                                for (auto& item : incoming) item.socket->close();
                                incoming.clear();
                                server->stop();
                                state.status = "LAN match";
                            }
                            continue;
                        }
                        if (type == "ping") { send(peer, {{"type", "pong"}}); continue; }
                        if (type == "pong") continue;
                        if (hosting) {
                            if (type == "leave") finish(1);
                            else if (type == "move" || type == "cancel") apply(message, 1, remoteAck);
                            else if (type == "recorded") {
                                require(finished && offset == replay.size(), "Unexpected recording acknowledgement");
                                received = true;
                            } else throw std::runtime_error("Unexpected LAN player packet");
                        } else if (type == "state") {
                            int rows = message.at("rows").get<int>(), cols = message.at("cols").get<int>();
                            std::istringstream input(message.at("view").get<std::string>());
                            auto view = Protocol::readObservation(input, {1, rows, cols}); input >> std::ws;
                            require(view.has_value() && input.eof(), "Invalid LAN observation");
                            int result = message.at("result").get<int>();
                            require(result >= 0 && result <= 3 && view->tick >= (*state.views)[1].tick, "Invalid LAN result or tick");
                            view->result = static_cast<Phases>(result);
                            auto views = std::make_shared<std::array<Observation, 3>>(); (*views)[1] = *view; state.views = views;
                            state.state = result == 0 ? MatchState::Playing : MatchState::Finishing;
                            if (result && !finished) finished = now;
                            const auto& queue = message.at("queued");
                            require(queue.is_array() && queue.size() <= Capacity, "Invalid LAN queue");
                            state.queued[1].clear();
                            for (auto& action : queue) state.queued[1].push_back(decode(action));
                            localAck = message.at("ack").get<std::uint64_t>();
                        } else if (type == "replay") {
                            require(finished && !received, "Unexpected LAN recording");
                            appendBytes(replay, message);
                            if (message.at("last").get<bool>()) {
                                Replay check;
                                std::string error;
                                require(check.loadBytes(replay, error), "Invalid LAN recording");
                                persist(replay, state); saved = received = true;
                                send(peer, {{"type", "recorded"}});
                            }
                        } else if (type == "closed") {
                            require(finished && received, "Room closed before its recording arrived");
                            mail->quit = true;
                        } else throw std::runtime_error("Unexpected LAN host packet");
                    }
                    std::erase_if(incoming, [&](const Incoming& item) {
                        if (now - item.arrived > 3s) item.socket->close();
                        return item.socket->isClosed();
                    });
                    if (!paired) {
                        if (peer && (peer->isClosed() || now - attempt > 5s)) {
                            peer->forceClose(); peer.reset(); host.clear(); hosting = helloSent = false; retry = now + 500ms;
                            if (proxy) throw std::runtime_error("LAN connection through PROXY failed; no direct fallback was attempted");
                        }
                        if (peer && !hosting && peer->isOpen() && !helloSent) {
                            send(peer, {{"type", "hello"}, {"id", id}, {"target", target}, {"room", config.room}, {"name", config.username}});
                            helloSent = true;
                        }
                        if (!peer && now >= retry && now - began >= 1s) {
                            if (!config.room.empty() && std::any_of(discovery.peers().begin(), discovery.peers().end(), [](const Peer& p) { return !p.host.empty(); }))
                                throw std::runtime_error("Room is full");
                            std::vector<std::string> available{id};
                            bool reserved = false;
                            for (auto& remote : discovery.peers()) {
                                // Private reservations remain in the election while their handshake is in flight.
                                if (remote.available || !config.room.empty()) available.push_back(remote.id);
                                reserved |= !remote.available;
                            }
                            std::sort(available.begin(), available.end());
                            auto rank = static_cast<std::size_t>(std::find(available.begin(), available.end(), id) - available.begin());
                            // Adjacent IDs pair; only the lower ID initiates. A pending socket reserves this window.
                            if (rank % 2 == 0 && rank + 1 < available.size() && (config.room.empty() || (rank == 0 && !reserved))) {
                                target = available[rank + 1];
                                auto remote = std::find_if(discovery.peers().begin(), discovery.peers().end(), [&](const Peer& p) { return p.id == target; });
                                rtc::WebSocket::Configuration transport;
                                transport.proxyServer = proxy; transport.connectionTimeout = 3s;
                                transport.pingInterval = 5s; transport.maxMessageSize = packetLimit;
                                peer = std::make_shared<rtc::WebSocket>(transport); listen(peer, mail);
                                peer->open("ws://" + remote->address + ":" + std::to_string(remote->port));
                                attempt = last = now; hosting = helloSent = false;
                            }
                        }
                        state.state = peer ? MatchState::Connecting : MatchState::Waiting;
                        publish(state);
                        std::this_thread::sleep_for(10ms);
                        continue;
                    }

                    for (auto& command : commands) {
                        if (finished || leaving) break;
                        if (hosting) apply(command, 0, localAck);
                        else send(peer, command);
                    }
                    if (hosting) {
                        if (!peer->isOpen()) finish(1);
                        auto local = match.snapshot();
                        state.views = local.views; state.queued = local.queued;
                        state.state = local.running() ? MatchState::Playing : MatchState::Finishing;
                        if (local.state == MatchState::Finished && !finished) finished = now;
                        if (peer->isOpen() && (sentViews != local.views || sentAck != remoteAck)) {
                            auto view = (*local.views)[1];
                            auto result = view.result; view.result = Phases::Ongoing;
                            std::ostringstream observation;
                            require(Protocol::writeObservation(observation, view), "Cannot encode LAN observation");
                            Json queue = Json::array();
                            for (auto& action : local.queued[1]) queue.push_back(encode(action));
                            send(peer, {{"type", "state"}, {"rows", view.rows}, {"cols", view.cols}, {"view", observation.str()},
                                        {"result", static_cast<int>(result)}, {"queued", queue}, {"ack", remoteAck}});
                            sentViews = local.views; sentAck = remoteAck;
                        }
                        if (finished && local.completed && replay.empty()) {
                            auto record = *local.completed; record.names = state.names;
                            replay = encodeReplay(record);
                            require(replay.size() <= recordingLimit, "LAN recording exceeds transfer limit");
                            persist(replay, state); saved = true;
                        }
                        if (finished && peer->isOpen() && offset < replay.size() && peer->bufferedAmount() < 65536) {
                            auto count = std::min(std::size_t{8192}, replay.size() - offset);
                            send(peer, {{"type", "replay"}, {"offset", offset}, {"data", hex(std::string_view(replay).substr(offset, count))},
                                        {"last", offset + count == replay.size()}});
                            offset += count;
                        }
                        if (finished && (received || !peer->isOpen() || now - *finished > 4s)) {
                            if (peer->isOpen()) send(peer, {{"type", "closed"}});
                            break;
                        }
                    } else if (!peer->isOpen() && !mail->quit) {
                        require(received, "LAN host disconnected before the final recording arrived");
                        break;
                    }
                    if (state.running() && !finished && !leaving && !config.command.empty()) {
                        const auto& view = (*state.views)[state.player];
                        if (view.rows > 0) {
                            try {
                                if (!agentStarted) {
                                    require(agent.start(config.command, {state.player, view.rows, view.cols}), "Cannot start LAN agent");
                                    agentStarted = true;
                                    requested = view.tick;
                                    replied = false;
                                    agent.request(view);
                                } else if (requested != view.tick) {
                                    requested = view.tick;
                                    replied = false;
                                    agent.request(view);
                                }
                                require(agent.error().empty(), "LAN agent stopped or returned an invalid action");
                                if (!replied) if (auto action = agent.reply(requested, now)) {
                                    replied = true;
                                    if (action->type == ActionType::Move)
                                        command({{"type", "move"}, {"tick", requested}, {"action", encode(*action)}});
                                }
                            } catch (const std::exception& error) {
                                state.error = error.what();
                                agent.stop();
                                agentStarted = false;
                                leaving = now;
                                if (hosting) finish(0);
                                else send(peer, {{"type", "leave"}});
                            }
                        }
                    }
                    if (finished && agentStarted) { agent.stop(); agentStarted = false; }
                    if (now - last > 20s) throw std::runtime_error("LAN peer timed out");
                    if (finished && !hosting && now - *finished > 6s) throw std::runtime_error("LAN recording transfer timed out");
                    if (now - heartbeat > 5s && peer->isOpen() && !mail->quit) { send(peer, {{"type", "ping"}}); heartbeat = now; }
                    publish(state, localAck);
                    std::this_thread::sleep_for(10ms);
                }
            } catch (const std::exception& error) { state.error = error.what(); }
            if (hosting && paired) {
                match.stop(state.error.empty() ? 0 : 1);
                auto local = match.snapshot(); state.views = local.views;
                if (!saved && local.completed) {
                    try { auto record = *local.completed; record.names = state.names; persist(encodeReplay(record), state); }
                    catch (const std::exception& error) { state.error = error.what(); }
                }
            }
            mail->quit = true;
            if (server) server->stop();
            if (peer) peer->close();
            for (auto& item : incoming) item.socket->close();
            agent.stop();
            {
                std::lock_guard lock(mail->mutex);
                mail->packets.clear(); mail->accepted.clear(); mail->commands.clear();
            }
            state.state = MatchState::Finished;
            auto result = state.player >= 0 ? (*state.views)[state.player].result : Phases::Ongoing;
            state.status = result == Phases::RedWin ? "Red wins" : result == Phases::BlueWin ? "Blue wins" : "LAN room closed";
            publish(state); open = false;
        }
    };

    LanMatch::LanMatch() : session(std::make_unique<Session>()) {}
    LanMatch::~LanMatch() = default;
    bool LanMatch::start(const LanConfig& config) {
        if (opened()) return false;
        if (session->worker.joinable()) session->worker.join();
        std::string error;
        if (!savePending(config.directory, error)) {
            auto state = snapshot(); state.error = error; session->publish(state); return false;
        }
        session = std::make_unique<Session>();
        try {
            require(nameValid(config.username), "LAN username must contain 1-64 printable ASCII characters");
            require(config.room.empty() || nameValid(config.room), "Room ID must contain at most 64 printable ASCII characters");
            require(config.directory.empty() || replayDirectory(config.directory, error), "Recording directory is unavailable");
            require(config.command.empty() || StrategyProcess::available(config.command), "Cannot open LAN player program");
            httpProxy(config.proxy);
            session->config = config; session->open = true;
            MatchSnapshot state; state.state = MatchState::Connecting; state.status = "Searching on LAN"; session->publish(state);
            session->worker = std::thread([this] { session->run(); });
            return true;
        } catch (const std::exception& error) {
            session->open = false; MatchSnapshot state; state.error = error.what(); session->publish(state); return false;
        }
    }
    void LanMatch::leave() { session->mail->leaving = true; }
    void LanMatch::stop() { leave(); }
    bool LanMatch::opened() const { return session->open; }
    MatchSnapshot LanMatch::snapshot() const { return session->snapshot(); }
    bool LanMatch::savePending(const std::filesystem::path& directory, std::string& error) {
        std::lock_guard lock(session->mutex);
        error.clear();
        if (session->unsaved.empty()) return true;
        if (opened()) { error = "Wait for the LAN recording transfer to finish before retrying RD"; return false; }
        if (directory.empty()) { error = "Set RD to save the pending LAN recording"; return false; }
        if (!saveReplayBytes(session->unsaved, directory, error)) return false;
        session->unsaved.clear(); session->published.error.clear();
        return true;
    }
    bool LanMatch::enqueue(int player, const Action& action) {
        auto state = snapshot();
        if (!opened() || player < 0 || player > 1 || player != state.player || !state.running() || !session->config.command.empty()) return false;
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
        if (!opened() || player < 0 || player > 1 || player != state.player || !state.running() || state.queued[player].empty()) return std::nullopt;
        auto removed = all ? state.queued[player].front() : state.queued[player].back();
        if (!session->command({{"type", "cancel"}, {"all", all}})) return std::nullopt;
        return removed;
    }
}
