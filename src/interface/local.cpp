// local session: combine submitted moves and external strategy replies into timed engine steps.
#include "local.hpp"

namespace NEBULA {
    LocalMatch::~LocalMatch() { stop(); }

    bool LocalMatch::start(const std::array<std::string, 2>& commands, std::uint32_t seed, int milliseconds,
                           const std::filesystem::path& destination) {
        std::unique_lock lock(mutex);
        if (phase == MatchState::Active) return false;
        lock.unlock();
        if (worker.joinable()) worker.join();
        lock.lock();
        failure.clear();
        // retain an unsaved game until a save succeeds, including when reset is requested after a write failure.
        if (recording) {
            if (!destination.empty()) directory = destination;
            if (!saveRecording()) { publish(); return false; }
        }
        if (!destination.empty() && !replayDirectory(destination, failure)) { publish(); return false; }
        if (milliseconds <= 0) { failure = "Half-turn milliseconds must be positive"; publish(); return false; }
        Engine next;
        if (!next.reset(0, 0, seed)) { failure = "Map generation failed"; publish(); return false; }
        std::array<std::unique_ptr<StrategyProcess>, 2> players;
        for (int player = 0; player < 2; ++player) {
            if (commands[player].empty()) continue;
            players[player] = std::make_unique<StrategyProcess>();
            auto view = *next.observe(player);
            Protocol::Init init{player, view.rows, view.cols};
            if (!players[player]->start(commands[player], init)) {
                failure = std::string(player == 0 ? "Red: " : "Blue: ") + players[player]->error();
                publish();
                return false;
            }
        }
        engine = std::move(next);
        directory = destination;
        saved.clear();
        if (!directory.empty()) recording.emplace(*engine.snapshot());
        strategies = std::move(players);
        for (auto& input : inputs) input.clear();
        phase = MatchState::Active;
        clock.interval = std::chrono::milliseconds(milliseconds);
        clock.reset();
        updateViews();
        request();
        clock.resume();
        publish();
        worker = std::thread([this] { run(); });
        return true;
    }

    void LocalMatch::stop() {
        {
            std::lock_guard lock(mutex);
            finish();
        }
        if (worker.joinable()) worker.join();
    }

    // the session lock covers all state changes, including manual stepping and process shutdown.
    void LocalMatch::finish() {
        clock.pause();
        if (phase == MatchState::Active) phase = MatchState::Finished;
        for (auto& input : inputs) input.clear();
        saveRecording();
        publish();
        // wake the paused worker when manual stepping ends the match, so a later start() can join it.
        changed.notify_all();
        for (auto& strategy : strategies) if (strategy) strategy->stop();
    }

    bool LocalMatch::saveRecording() {
        saveError.clear();
        if (!recording) return true;
        auto path = saveReplay(*recording, directory, saveError);
        if (!path) return false;
        saved = *path;
        recording.reset();
        return true;
    }

    void LocalMatch::pause() {
        std::lock_guard lock(mutex);
        clock.pause();
        publish();
        changed.notify_all();
    }

    void LocalMatch::resume() {
        std::lock_guard lock(mutex);
        if (phase == MatchState::Active) clock.resume();
        publish();
        changed.notify_all();
    }

    bool LocalMatch::setInterval(int milliseconds) {
        if (milliseconds <= 0) return false;
        std::lock_guard lock(mutex);
        bool running = clock.running();
        clock.interval = std::chrono::milliseconds(milliseconds);
        clock.reset();
        if (running) clock.resume();
        changed.notify_all();
        return true;
    }

    LocalSnapshot LocalMatch::snapshot() const {
        std::lock_guard lock(snapshotMutex);
        return published;
    }

    bool LocalMatch::enqueue(int player, const Action& action) {
        std::lock_guard lock(mutex);
        if (phase != MatchState::Active || player < 0 || player > 1 || strategies[player]) return false;
        const auto& view = (*views)[player];
        if (action.type != ActionType::Move || action.row < 0 || action.row >= view.rows ||
            action.col < 0 || action.col >= view.cols) return false;
        int row = action.row, col = action.col;
        switch (action.direction) {
            case Direction::Up: --row; break;
            case Direction::Down: ++row; break;
            case Direction::Left: --col; break;
            case Direction::Right: ++col; break;
            default: return false;
        }
        if (row < 0 || row >= view.rows || col < 0 || col >= view.cols) return false;
        // bound stored input while allowing a route across the whole board.
        if (inputs[player].size() >= Capacity) return false;
        inputs[player].push_back({action, Clock::Source::now()});
        publish();
        return true;
    }

    std::optional<Action> LocalMatch::cancel(int player, bool all) {
        std::lock_guard lock(mutex);
        if (player < 0 || player > 1 || inputs[player].empty()) return std::nullopt;
        auto& input = inputs[player];
        Action removed = all ? input.front().action : input.back().action;
        if (all) input.clear();
        else input.pop_back();
        publish();
        return removed;
    }

    // callers hold the session lock. the display lock covers only this already-prepared value.
    void LocalMatch::publish() {
        LocalSnapshot next;
        next.views = views;
        next.state = phase;
        next.running = clock.running();
        next.error = failure;
        if (!saveError.empty()) next.error += (next.error.empty() ? "" : ". ") + saveError;
        next.saved = saved;
        next.unsaved = recording.has_value() && phase == MatchState::Finished;
        for (int player = 0; player < 2; ++player) {
            for (const auto& pending : inputs[player]) next.queued[player].push_back(pending.action);
        }
        std::lock_guard lock(snapshotMutex);
        published = std::move(next);
    }

    void LocalMatch::request() {
        for (int player = 0; player < 2; ++player) {
            if (strategies[player]) strategies[player]->request((*views)[player]);
        }
    }

    // window event processing can stall while dragging. this thread keeps half-turn deadlines independent of drawing.
    void LocalMatch::run() {
        std::unique_lock lock(mutex);
        while (phase == MatchState::Active) {
            changed.wait(lock, [&] { return phase != MatchState::Active || clock.running(); });
            if (phase != MatchState::Active) break;
            auto deadline = clock.expires();
            if (changed.wait_until(lock, deadline, [&] {
                return phase != MatchState::Active || !clock.running() || clock.expires() != deadline;
            })) continue;
            clock.consume();
            settle(deadline);
        }
    }

    bool LocalMatch::advance() {
        std::lock_guard lock(mutex);
        if (phase != MatchState::Active || clock.running()) return false;
        clock.reset();
        return settle(Clock::Source::now());
    }

    bool LocalMatch::settle(Clock::Time cutoff) {
        if (phase != MatchState::Active) return false;
        auto tick = (*views)[0].tick;
        std::array<Action, 2> actions;
        // an absent or late reply becomes Pass. each reply is tied to the observation it was computed from.
        for (int player = 0; player < 2; ++player) {
            if (!strategies[player]) {
                auto& input = inputs[player];
                // an input received after this deadline belongs to a later half-turn.
                if (!input.empty() && input.front().received <= cutoff) {
                    actions[player] = input.front().action;
                    input.pop_front();
                }
                continue;
            }
            std::string error = strategies[player]->error();
            if (!error.empty()) {
                failure = std::string(player == 0 ? "Red: " : "Blue: ") + error;
                finish();
                return false;
            }
            actions[player] = strategies[player]->action(tick, cutoff);
        }
        engine.step(actions);
        if (recording) recording->append(*engine.snapshot(), actions);
        updateViews();
        if ((*views)[0].result != Phases::Ongoing) finish();
        else { publish(); request(); }
        return true;
    }

    // prepare visibility once per position. full-board inspection shares public totals with the red observation.
    void LocalMatch::updateViews() {
        views = std::make_shared<const std::array<Observation, 3>>(matchViews(*engine.snapshot()));
    }

    std::array<Observation, 3> matchViews(const States& state) {
        std::array<Observation, 3> next{observe(state, 0), observe(state, 1)};
        next[2] = next[0];
        auto& observation = next[2];
        for (int row = 0; row < observation.rows; ++row) {
            for (int col = 0; col < observation.cols; ++col) {
                const Cell& cell = state.board.at(row, col);
                ViewCell& shown = observation.cells[row * observation.cols + col];
                switch (cell.terrain) {
                    case Terrain::Plain: shown.terrain = ViewTerrain::Plain; break;
                    case Terrain::Mountain: shown.terrain = ViewTerrain::Mountain; break;
                    case Terrain::City: shown.terrain = ViewTerrain::City; break;
                    case Terrain::General: shown.terrain = ViewTerrain::General; break;
                }
                shown.owner = cell.owner;
                shown.army = cell.army;
            }
        }
        return next;
    }
}
