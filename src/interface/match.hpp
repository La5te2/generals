#pragma once

#include "agents/builtin.hpp"
#include "engine/engine.hpp"

namespace NEBULA {
    enum class Participant { Human, Builtin, External };
    enum class MatchState { Empty, Active, Finished };

    // owns one local game. input, drawing and wall-clock scheduling belong to its caller.
    class Match {
    public:
        bool start(std::array<Participant, 2> participants, std::uint32_t seed);
        void stop();
        bool advance();
        Observation view(int perspective) const;
        MatchState state() const { return phase; }

    private:
        Engine engine;
        MatchState phase = MatchState::Empty;
        std::array<std::mt19937, 2> random;
    };
}
