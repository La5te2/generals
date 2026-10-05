#include "match.hpp"

namespace NEBULA {
    bool Match::start(std::array<Participant, 2> participants, std::uint32_t seed) {
        if (phase == MatchState::Active) return false;
        if (participants[0] != Participant::Builtin || participants[1] != Participant::Builtin) return false;
        if (!engine.reset(0, 0, seed)) return false;
        std::mt19937 seeds(seed);
        for (int player = 0; player < 2; ++player) {
            random[player].seed(seeds());
        }
        phase = MatchState::Active;
        return true;
    }

    void Match::stop() {
        if (phase == MatchState::Active) phase = MatchState::Finished;
    }

    bool Match::advance() {
        if (phase != MatchState::Active) return false;
        std::array<Action, 2> actions;
        // both strategies see the position before either action is executed.
        for (int player = 0; player < 2; ++player) {
            actions[player] = Agents::builtin(*engine.observe(player), random[player]);
        }
        engine.step(actions);
        if (engine.snapshot()->result != Phases::Ongoing) phase = MatchState::Finished;
        return true;
    }

    Observation Match::view(int perspective) const {
        if (phase == MatchState::Empty) return {};
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
