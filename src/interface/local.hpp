// local session: collect external strategy replies and advance the engine at half-turn deadlines.
#pragma once

#include "clock.hpp"
#include "engine/engine.hpp"
#include "process.hpp"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace NEBULA {
    enum class LocalState { Empty, Active, Finished };

    struct LocalSnapshot {
        // all three perspectives belong to the same completed half-turn. readers keep them alive while drawing.
        std::shared_ptr<const std::array<Observation, 3>> views;
        LocalState state = LocalState::Empty;
        bool running = false;
        std::string error;
    };

    class LocalMatch {
    public:
        ~LocalMatch();
        bool start(const std::array<std::string, 2>& commands, std::uint32_t seed, int milliseconds = 500);
        void stop();
        void pause();
        void resume();
        // change future half-turn lengths while preserving the current deadline or paused remainder.
        bool setInterval(int milliseconds);
        // manually settle one half-turn while paused, using replies available at the time of this call.
        bool advance();
        // read the latest published board and status, independently of rule updates and strategy communication.
        LocalSnapshot snapshot() const;

    private:
        void run();
        bool settle(Clock::Time cutoff);
        void request();
        void finish();
        void updateViews();
        void publish();

        mutable std::mutex mutex;
        std::condition_variable changed;
        std::thread worker;
        Engine engine;
        Clock clock;
        LocalState phase = LocalState::Empty;
        std::array<std::unique_ptr<StrategyProcess>, 2> strategies;
        std::string failure;
        std::shared_ptr<const std::array<Observation, 3>> views = std::make_shared<const std::array<Observation, 3>>();
        // hold this separate lock only while exchanging the small snapshot, never while computing or stopping processes.
        mutable std::mutex snapshotMutex;
        LocalSnapshot published{views};
    };
}
