// local session: collect external strategy replies and advance the engine at half-turn deadlines.
#pragma once

#include "clock.hpp"
#include "engine/engine.hpp"
#include "process.hpp"
#include <condition_variable>
#include <mutex>
#include <thread>

namespace NEBULA {
    enum class LocalState { Empty, Active, Finished };

    class LocalMatch {
    public:
        ~LocalMatch();
        bool start(const std::array<std::string, 2>& commands, std::uint32_t seed, int milliseconds = 500);
        void stop();
        void pause();
        void resume();
        // change future half-turn lengths while preserving the current deadline or paused remainder.
        bool setInterval(int milliseconds);
        bool running() const;
        // manually settle one half-turn while paused, using replies available at the time of this call.
        bool advance();
        Observation view(int perspective) const;
        LocalState state() const;
        std::string error() const;

    private:
        void run();
        bool settle(Clock::Time cutoff);
        void request();
        void finish();

        mutable std::mutex mutex;
        std::condition_variable changed;
        std::thread worker;
        Engine engine;
        Clock clock;
        LocalState phase = LocalState::Empty;
        std::array<std::unique_ptr<StrategyProcess>, 2> strategies;
        std::string failure;
    };
}
