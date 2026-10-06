// external agent interface: let the game session start and stop an agent process, send observations and collect timely actions.
#pragma once

#include "engine/protocol.hpp"
#include <chrono>
#include <memory>
#include <string>
#include <string_view>

namespace NEBULA {
    // one agent process controls one player during a game.
    // its stdin receives the player's index and board dimensions first, followed by observations.
    // the agent replies to each observation with one action on stdout, and writes diagnostics to stderr.
    class StrategyProcess {
    public:
        using Time = std::chrono::steady_clock::time_point;
        StrategyProcess();
        ~StrategyProcess();
        StrategyProcess(const StrategyProcess&) = delete;
        StrategyProcess& operator=(const StrategyProcess&) = delete;

        // command contains a program and arguments. quotes group spaces, backslashes remain literal.
        // programs run directly, with the application's working directory and inherited stderr.
        bool start(std::string_view command, const Protocol::Init& init);
        // close the agent's stdin to signal game end through EOF. an agent that stays blocked is terminated.
        void stop();
        // while the agent is calculating an action, retain only the newest requested observation.
        // send that observation after its current reply arrives, so the next observed tick may skip several half-turns.
        void request(const Observation& view);
        // only a matching reply received by cutoff can supply this half-turn's action.
        Action action(std::uint64_t tick, Time cutoff) const;
        // online sessions send a reply as soon as it arrives, including an explicit Pass.
        std::optional<Action> reply(std::uint64_t tick, Time cutoff) const;
        static bool available(std::string_view command);
        std::string error() const;

    private:
        struct Process;
        std::unique_ptr<Process> process;
    };
}
