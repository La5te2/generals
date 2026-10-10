// local session: combine submitted moves and external agent replies into timed engine steps.
#include "local.hpp"

namespace NEBULA {
    LocalMatch::~LocalMatch() { stop(); }

    bool LocalMatch::start(const std::array<std::string, 2>& commands, std::uint32_t seed, int milliseconds,
                           const std::filesystem::path& destination, bool retainRecording) {
        std::unique_lock lock(mutex);
        if (phase == Phase::Active) return false;
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
        completed.reset();
        if (!directory.empty() || retainRecording) recording.emplace(*engine.snapshot());
        strategies = std::move(players);
        for (auto& input : inputs) input.clear();
        phase = Phase::Active;
        clock.interval = std::chrono::milliseconds(milliseconds);
        clock.reset();
        updateViews();
        request();
        clock.resume();
        publish();
        worker = std::thread([this] { run(); });
        return true;
    }

    void LocalMatch::stop(int surrender) {
        {
            std::lock_guard lock(mutex);
            int human = !strategies[0] && strategies[1] ? 0 : strategies[0] && !strategies[1] ? 1 : -1;
            finish(surrender == 0 || surrender == 1 ? surrender : human);
        }
        if (worker.joinable()) worker.join();
    }

    // the session lock covers all state changes, including manual stepping and process shutdown.
    void LocalMatch::finish(int surrender) {
        clock.pause();
        if (phase == Phase::Active) {
            auto state = engine.snapshot();
            if (state->result == Phases::Ongoing) {
                // stop surrenders the human player. two programs use the timeout comparison instead.
                // finish that half-turn with two passes so the recorded surrender has the same final position.
                auto score = engine.observe(0);
                int winner = score->armies[0] > score->armies[1] ||
                    (score->armies[0] == score->armies[1] && score->land[0] > score->land[1]) ? 0 : 1;
                if (surrender >= 0) winner = 1 - surrender;
                engine.step({});
                state = engine.snapshot();
                state->result = winner == 0 ? Phases::RedWin : Phases::BlueWin;
                engine.load(*state);
                if (recording) {
                    recording->append(*state, {});
                    recording->surrendered = 1 - winner;
                }
                updateViews();
            }
            phase = Phase::Finished;
            if (recording) completed = std::make_shared<const Recording>(*recording);
        }
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
        if (directory.empty()) { recording.reset(); return true; }
        auto path = saveReplay(*recording, directory, saveError);
        if (!path) return false;
        saved = *path;
        recording.reset();
        return true;
    }

    bool LocalMatch::savePending(const std::filesystem::path& destination, std::string& error) {
        std::lock_guard lock(mutex);
        error.clear();
        if (phase != Phase::Finished || !recording) return true;
        if (destination.empty()) { error = "Set RD to save the pending local recording"; return false; }
        directory = destination;
        bool success = saveRecording();
        error = saveError;
        publish();
        return success;
    }

    void LocalMatch::pause() {
        std::lock_guard lock(mutex);
        clock.pause();
        publish();
        changed.notify_all();
    }

    void LocalMatch::resume() {
        std::lock_guard lock(mutex);
        if (phase == Phase::Active) clock.resume();
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
        if (phase != Phase::Active || player < 0 || player > 1 || strategies[player]) return false;
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
        next.state = phase == Phase::Active ? (clock.running() ? MatchState::Playing : MatchState::Paused) :
                     phase == Phase::Finished ? MatchState::Finished : MatchState::Empty;
        next.error = failure;
        if (!saveError.empty()) next.error += (next.error.empty() ? "" : ". ") + saveError;
        next.saved = saved;
        next.completed = completed;
        next.unsaved = recording.has_value() && phase == Phase::Finished;
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
        while (phase == Phase::Active) {
            changed.wait(lock, [&] { return phase != Phase::Active || clock.running(); });
            if (phase != Phase::Active) break;
            auto deadline = clock.expires();
            if (changed.wait_until(lock, deadline, [&] {
                return phase != Phase::Active || !clock.running() || clock.expires() != deadline;
            })) continue;
            clock.consume();
            settle(deadline);
        }
    }

    bool LocalMatch::advance() {
        std::lock_guard lock(mutex);
        if (phase != Phase::Active || clock.running()) return false;
        clock.reset();
        return settle(Clock::Source::now());
    }

    bool LocalMatch::settle(Clock::Time cutoff) {
        if (phase != Phase::Active) return false;
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

}
