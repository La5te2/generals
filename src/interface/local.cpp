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
        if (milliseconds <= 0) { failure = "Half-turn milliseconds must be positive"; publish(); return false; }
        Engine next;
        if (!next.reset(0, 0, seed)) { failure = "Map generation failed"; publish(); return false; }
        std::array<std::unique_ptr<StrategyProcess>, 2> players;
        for (int player = 0; player < 2; ++player) {
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
        strategies = std::move(players);
        phase = LocalState::Active;
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
        if (phase == LocalState::Active) phase = LocalState::Finished;
        publish();
        // wake the paused worker when manual stepping ends the match, so a later start() can join it.
        changed.notify_all();
        for (auto& strategy : strategies) if (strategy) strategy->stop();
    }

    void LocalMatch::pause() {
        std::lock_guard lock(mutex);
        clock.pause();
        publish();
        changed.notify_all();
    }

    void LocalMatch::resume() {
        std::lock_guard lock(mutex);
        if (phase == LocalState::Active) clock.resume();
        publish();
        changed.notify_all();
    }

    bool LocalMatch::setInterval(int milliseconds) {
        if (milliseconds <= 0) return false;
        std::lock_guard lock(mutex);
        clock.interval = std::chrono::milliseconds(milliseconds);
        return true;
    }

    LocalSnapshot LocalMatch::snapshot() const {
        std::lock_guard lock(snapshotMutex);
        return published;
    }

    // callers hold the session lock. the display lock covers only this already-prepared value.
    void LocalMatch::publish() {
        LocalSnapshot next{views, phase, clock.running(), failure};
        std::lock_guard lock(snapshotMutex);
        published = std::move(next);
    }

    void LocalMatch::request() {
        for (int player = 0; player < 2; ++player) strategies[player]->request((*views)[player]);
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
        auto tick = (*views)[0].tick;
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
        updateViews();
        if ((*views)[0].result != Phases::Ongoing) finish();
        else { publish(); request(); }
        return true;
    }

    // prepare visibility once per position. full-board inspection shares public totals with the red observation.
    void LocalMatch::updateViews() {
        auto state = engine.snapshot();
        auto next = std::make_shared<std::array<Observation, 3>>();
        (*next)[0] = observe(*state, 0);
        (*next)[1] = observe(*state, 1);
        (*next)[2] = (*next)[0];
        auto& observation = (*next)[2];
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
        views = std::move(next);
    }
}
