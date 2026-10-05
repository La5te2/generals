// connects external strategies/agents to the rule engine and settle their actions at half-turn deadlines.
#include "local.hpp"

namespace NEBULA {
    LocalMatch::~LocalMatch() { stop(); }

    bool LocalMatch::start(const std::array<std::string, 2>& commands, std::uint32_t seed, int milliseconds) {
        std::unique_lock lock(mutex);
        if (phase == LocalState::Active) return false;
        lock.unlock();
        if (worker.joinable()) worker.join();
        lock.lock();
        failure.clear();
        if (milliseconds <= 0) { failure = "Half-turn milliseconds must be positive"; return false; }
        Engine next;
        if (!next.reset(0, 0, seed)) { failure = "Map generation failed"; return false; }
        std::array<std::unique_ptr<StrategyProcess>, 2> players;
        for (int player = 0; player < 2; ++player) {
            players[player] = std::make_unique<StrategyProcess>();
            auto view = *next.observe(player);
            Protocol::Init init{player, view.rows, view.cols};
            if (!players[player]->start(commands[player], init)) {
                failure = std::string(player == 0 ? "Red: " : "Blue: ") + players[player]->error();
                return false;
            }
        }
        engine = std::move(next);
        strategies = std::move(players);
        phase = LocalState::Active;
        clock.interval = std::chrono::milliseconds(milliseconds);
        clock.reset();
        request();
        clock.resume();
        worker = std::thread([this] { run(); });
        return true;
    }

    void LocalMatch::stop() {
        {
            std::lock_guard lock(mutex);
            finish();
        }
        changed.notify_all();
        if (worker.joinable()) worker.join();
    }

    // the session lock covers all state changes, including manual stepping and process shutdown.
    void LocalMatch::finish() {
        clock.pause();
        if (phase == LocalState::Active) phase = LocalState::Finished;
        for (auto& strategy : strategies) if (strategy) strategy->stop();
    }

    void LocalMatch::pause() {
        std::lock_guard lock(mutex);
        clock.pause();
        changed.notify_all();
    }

    void LocalMatch::resume() {
        std::lock_guard lock(mutex);
        if (phase == LocalState::Active) clock.resume();
        changed.notify_all();
    }

    bool LocalMatch::setInterval(int milliseconds) {
        if (milliseconds <= 0) return false;
        std::lock_guard lock(mutex);
        clock.interval = std::chrono::milliseconds(milliseconds);
        return true;
    }

    bool LocalMatch::running() const { std::lock_guard lock(mutex); return clock.running(); }
    LocalState LocalMatch::state() const { std::lock_guard lock(mutex); return phase; }
    std::string LocalMatch::error() const { std::lock_guard lock(mutex); return failure; }

    void LocalMatch::request() {
        for (int player = 0; player < 2; ++player) strategies[player]->request(*engine.observe(player));
    }

    // window event processing can stall while dragging. this thread keeps half-turn deadlines independent of drawing.
    void LocalMatch::run() {
        std::unique_lock lock(mutex);
        while (phase == LocalState::Active) {
            changed.wait(lock, [&] { return phase != LocalState::Active || clock.running(); });
            if (phase != LocalState::Active) break;
            auto deadline = clock.expires();
            if (changed.wait_until(lock, deadline, [&] {
                return phase != LocalState::Active || !clock.running() || clock.expires() != deadline;
            })) continue;
            clock.consume();
            settle(deadline);
        }
    }

    bool LocalMatch::advance() {
        std::lock_guard lock(mutex);
        if (phase != LocalState::Active || clock.running()) return false;
        clock.reset();
        return settle(Clock::Source::now());
    }

    bool LocalMatch::settle(Clock::Time cutoff) {
        if (phase != LocalState::Active) return false;
        auto tick = engine.snapshot()->tick;
        std::array<Action, 2> actions;
        // an absent or late reply becomes Pass. each reply is tied to the observation it was computed from.
        for (int player = 0; player < 2; ++player) {
            std::string error = strategies[player]->error();
            if (!error.empty()) {
                failure = std::string(player == 0 ? "Red: " : "Blue: ") + error;
                finish();
                return false;
            }
            actions[player] = strategies[player]->action(tick, cutoff);
        }
        engine.step(actions);
        if (engine.snapshot()->result != Phases::Ongoing) finish();
        else request();
        return true;
    }

    Observation LocalMatch::view(int perspective) const {
        std::lock_guard lock(mutex);
        if (phase == LocalState::Empty) return {};
        if (perspective == 0 || perspective == 1) return *engine.observe(perspective);
        Observation observation = *engine.observe(0);
        auto state = engine.snapshot();
        for (int row = 0; row < observation.rows; ++row) {
            for (int col = 0; col < observation.cols; ++col) {
                const Cell& cell = state->board.at(row, col);
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
        return observation;
    }
}
