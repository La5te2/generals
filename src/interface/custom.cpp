// one outgoing WebSocket session to a custom server. input queues remain responsive while replies are in flight.
#include "custom.hpp"
#include "process.hpp"
#include "proxy.hpp"
#include "replay.hpp"
#include "server/wire.hpp"
#include <atomic>
#include <thread>

namespace NEBULA {
    namespace {
        using namespace Wire;
        using namespace std::chrono_literals;
        // empty selects the local server. otherwise require a WebSocket URL without credentials or whitespace.
        std::string endpoint(const std::string& address) {
            if (address.empty()) return "ws://127.0.0.1:42817";
            auto begin = address.starts_with("ws://") ? 5u : address.starts_with("wss://") ? 6u : 0u;
            require(begin && address.size() > begin && address.size() <= 2048 &&
                    address.find_first_of("# \t\r\n") == std::string::npos, "Address requires ws://host:port or wss://host/path");
            auto authority = std::string_view(address).substr(begin, address.find('/', begin) - begin);
            require(!authority.empty() && authority.find_first_of("@?") == std::string_view::npos, "Invalid server address");
            return address;
        }
    }
    struct ClientMatch::Session {
        mutable std::mutex mutex;
        MatchSnapshot published;
        std::thread worker;
        std::atomic<bool> open{false}, leaving{false};
        ClientConfig config;
        std::string unsaved;
        std::uint64_t sequence = 0;
        std::deque<Json> pending, commands;

        ~Session() { leaving = true; if (worker.joinable()) worker.join(); }
        MatchSnapshot snapshot() const { std::lock_guard lock(mutex); return published; }
        // reconcile server acknowledgements, then overlay unacknowledged input so route markers do not jump backwards.
        void publish(MatchSnapshot value, std::uint64_t ack = 0) {
            std::lock_guard lock(mutex);
            std::erase_if(pending, [&](const Json& item) { return item["id"].get<std::uint64_t>() <= ack; });
            if (value.running() && value.player >= 0) {
                auto& queue = value.queued[value.player];
                for (auto& item : pending) {
                    if (item["type"] == "move") queue.push_back(decode(item["action"]));
                    else if (item["all"].get<bool>()) queue.clear();
                    else if (!queue.empty()) queue.pop_back();
                }
            } else pending.clear();
            published = std::move(value);
        }
        // under one lock, assign an input id, update route feedback and queue the corresponding network command.
        bool command(Json value) {
            std::lock_guard lock(mutex);
            if (leaving || !published.running() || commands.size() >= Capacity || pending.size() >= Capacity) return false;
            value["id"] = ++sequence;
            pending.push_back(value);
            auto& queue = published.queued[published.player];
            if (value["type"] == "move") queue.push_back(decode(value["action"]));
            else if (value["all"].get<bool>()) queue.clear();
            else if (!queue.empty()) queue.pop_back();
            commands.push_back(std::move(value));
            return true;
        }
        // save the verified final replay when RD is enabled. retain its bytes and report the error if writing fails.
        void persist(const std::string& bytes, MatchSnapshot& state) {
            if (config.directory.empty()) return;
            std::string error;
            if (!saveReplayBytes(bytes, config.directory, error)) {
                std::lock_guard lock(mutex);
                unsaved = bytes; state.error = error;
            }
        }
        // own this connection and optional agent on the worker thread. the server remains authoritative for all moves.
        // validate the final replay against the last observation before exposing RED/BLUE/ALL, then publish completion.
        void run() {
            MatchSnapshot state;
            state.state = MatchState::Connecting; state.status = "Connecting to custom server";
            StrategyProcess agent;
            try {
                rtc::WebSocket::Configuration transport;
                transport.proxyServer = httpProxy(config.proxy); transport.connectionTimeout = 10s;
                transport.pingInterval = 5s; transport.maxMessageSize = packetLimit;
                Connection peer(std::make_shared<rtc::WebSocket>(transport));
                peer.socket->open(endpoint(config.address));
                bool joined = false, paired = false, received = false, closed = false, agentStarted = false, replied = false;
                std::uint64_t ack = 0, requested = 0;
                std::string replay;
                auto began = Time::now();
                std::optional<Time::time_point> finished, departure;
                publish(state);
                while (!closed) {
                    auto now = Time::now();
                    if (leaving && !departure) {
                        if (!joined) break;
                        peer.send({{"type", "leave"}}); departure = now;
                    }
                    require(!departure || now - *departure < 35s, "Server did not finish closing the room");
                    if (peer.socket->isOpen() && !joined) {
                        peer.send({{"type", "join"}, {"version", version}, {"name", config.username}, {"room", config.room}});
                        joined = true;
                    }
                    for (auto& text : peer.take()) {
                        auto message = parse(text);
                        if (peer.control(message)) continue;
                        auto type = message.at("type").get<std::string>();
                        if (type == "error") throw std::runtime_error(message.at("message").get<std::string>().substr(0, 256));
                        if (type == "waiting") {
                            require(joined && !paired, "Unexpected waiting packet");
                            state.state = MatchState::Waiting; state.status = "Waiting for an opponent";
                        } else if (type == "paired") {
                            require(joined && !paired, "Unexpected pairing packet");
                            int seat = message.at("player").get<int>();
                            require(seat == 0 || seat == 1, "Invalid player seat");
                            auto names = message.at("names").get<std::array<std::string, 2>>();
                            require(nameValid(names[0]) && nameValid(names[1]) && names[seat] == config.username, "Invalid player names");
                            state.player = seat; state.names = std::move(names); paired = true;
                            state.status = "Starting match";
                            peer.send({{"type", "ready"}});
                        } else if (type == "state") {
                            require(paired && !finished, "Unexpected board update");
                            int rows = message.at("rows").get<int>(), cols = message.at("cols").get<int>();
                            const auto& previous = (*state.views)[state.player];
                            require(previous.rows == 0 || (rows == previous.rows && cols == previous.cols), "Board dimensions changed");
                            std::istringstream input(message.at("view").get<std::string>());
                            auto view = Protocol::readObservation(input, {state.player, rows, cols}); input >> std::ws;
                            require(view.has_value() && input.eof(), "Invalid server observation");
                            int result = message.at("result").get<int>();
                            require(result >= 0 && result <= 3 && view->tick >= previous.tick, "Invalid result or observation tick");
                            view->result = static_cast<Phases>(result);
                            auto views = std::make_shared<std::array<Observation, 3>>(*state.views);
                            (*views)[state.player] = *view; state.views = views;
                            state.state = result ? MatchState::Finishing : MatchState::Playing;
                            state.status = result ? "Receiving final recording" : "Custom match";
                            if (result) finished = now;
                            const auto& queue = message.at("queued");
                            require(queue.is_array() && queue.size() <= Capacity, "Invalid input queue");
                            auto& actions = state.queued[state.player]; actions.clear();
                            for (auto& action : queue) actions.push_back(decode(action));
                            require(message.at("ack").is_number_unsigned(), "Invalid input acknowledgement");
                            auto nextAck = message["ack"].get<std::uint64_t>();
                            { std::lock_guard lock(mutex); require(nextAck >= ack && nextAck <= sequence, "Invalid input acknowledgement"); }
                            ack = nextAck;
                        } else if (type == "replay") {
                            require(finished && !received, "Unexpected recording");
                            append(replay, message);
                            if (message.at("last").get<bool>()) {
                                Replay check;
                                std::string error;
                                require(check.loadBytes(replay, error), "Invalid recording");
                                const auto& view = (*state.views)[state.player];
                                require(check.seek(check.length()) && check.state().tick == view.tick && check.state().result == view.result &&
                                        check.names() == state.names, "Recording differs from the final observation");
                                auto final = matchViews(check.state());
                                auto expected = final[state.player], observed = view;
                                expected.result = observed.result = Phases::Ongoing;
                                std::ostringstream a, b;
                                require(Protocol::writeObservation(a, expected) && Protocol::writeObservation(b, observed) && a.str() == b.str(),
                                        "Recording board differs from the final observation");
                                state.views = std::make_shared<const std::array<Observation, 3>>(std::move(final));
                                persist(replay, state); received = true;
                                peer.send({{"type", "recorded"}});
                            }
                        } else if (type == "closed") {
                            require(!paired || received, "Room closed before its recording arrived");
                            closed = true;
                        } else throw std::runtime_error("Unexpected server packet");
                    }
                    if (closed) break;
                    require(!peer.closed, config.proxy.empty() ? "Server connection closed" : "Server connection through PROXY failed; no direct fallback was attempted");
                    require(joined || now - began < 12s, "Server connection timed out");
                    if (!joined) { std::this_thread::sleep_for(10ms); continue; }
                    peer.pulse();
                    require(!finished || now - *finished < 35s, "Recording transfer timed out");
                    publish(state, ack);
                    if (state.running() && !leaving && !departure && !config.command.empty()) {
                        const auto& view = (*state.views)[state.player];
                        try {
                            if (!agentStarted) {
                                require(agent.start(config.command, {state.player, view.rows, view.cols}), "Cannot start agent");
                                agentStarted = true; requested = view.tick; replied = false; agent.request(view);
                            } else if (requested != view.tick) { requested = view.tick; replied = false; agent.request(view); }
                            require(agent.error().empty(), "Agent stopped or returned an invalid action");
                            if (!replied) if (auto action = agent.reply(requested, now)) {
                                replied = true;
                                if (action->type == ActionType::Move) command({{"type", "move"}, {"tick", requested}, {"action", encode(*action)}});
                            }
                        } catch (const std::exception& error) { state.error = error.what(); leaving = true; }
                    }
                    if ((finished || leaving) && agentStarted) { agent.stop(); agentStarted = false; }
                    std::deque<Json> outbound;
                    { std::lock_guard lock(mutex); outbound.swap(commands); }
                    if (state.running() && !leaving && !departure) for (auto& command : outbound) peer.send(command);
                    std::this_thread::sleep_for(10ms);
                }
            } catch (const std::exception& error) { state.error = error.what(); }
            agent.stop();
            state.state = MatchState::Finished;
            auto result = state.player >= 0 ? (*state.views)[state.player].result : Phases::Ongoing;
            state.status = result == Phases::RedWin ? "Red wins" : result == Phases::BlueWin ? "Blue wins" :
                           result == Phases::Draw ? "Draw" : "Server session closed";
            publish(state); open = false;
        }
    };

    ClientMatch::ClientMatch() : session(std::make_unique<Session>()) {}
    ClientMatch::~ClientMatch() = default;
    bool ClientMatch::start(const ClientConfig& config) {
        if (opened()) return false;
        if (session->worker.joinable()) session->worker.join();
        std::string error;
        if (!savePending(config.directory, error)) { auto state = snapshot(); state.error = error; session->publish(state); return false; }
        session = std::make_unique<Session>();
        try {
            require(nameValid(config.username), "Username must contain 1-64 printable ASCII characters");
            require(config.room.empty() || nameValid(config.room), "Room ID must contain at most 64 printable ASCII characters");
            require(config.directory.empty() || replayDirectory(config.directory, error), "Recording directory is unavailable");
            require(config.command.empty() || StrategyProcess::available(config.command), "Cannot open player program");
            endpoint(config.address); httpProxy(config.proxy);
            session->config = config; session->open = true;
            MatchSnapshot state; state.state = MatchState::Connecting; state.status = "Connecting to custom server"; session->publish(state);
            session->worker = std::thread([active = session.get()] { active->run(); });
            return true;
        } catch (const std::exception& error) {
            session->open = false; MatchSnapshot state; state.error = error.what(); session->publish(state); return false;
        }
    }
    void ClientMatch::leave() { session->leaving = true; }
    void ClientMatch::stop() { leave(); }
    bool ClientMatch::opened() const { return session->open; }
    MatchSnapshot ClientMatch::snapshot() const { return session->snapshot(); }
    bool ClientMatch::savePending(const std::filesystem::path& directory, std::string& error) {
        std::lock_guard lock(session->mutex);
        error.clear();
        if (session->unsaved.empty()) return true;
        if (opened()) { error = "Wait for the recording transfer to finish before retrying RD"; return false; }
        if (directory.empty()) { error = "Set RD to save the pending recording"; return false; }
        if (!saveReplayBytes(session->unsaved, directory, error)) return false;
        session->unsaved.clear(); session->published.error.clear();
        return true;
    }
    bool ClientMatch::enqueue(int player, const Action& action) {
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
    std::optional<Action> ClientMatch::cancel(int player, bool all) {
        auto state = snapshot();
        if (!opened() || player < 0 || player > 1 || player != state.player || !state.running() || state.queued[player].empty()) return std::nullopt;
        auto removed = all ? state.queued[player].front() : state.queued[player].back();
        if (!session->command({{"type", "cancel"}, {"all", all}})) return std::nullopt;
        return removed;
    }
}
