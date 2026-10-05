#pragma once

#include "actions.hpp"
#include "initial.hpp"
#include "observe.hpp"
#include "protocol.hpp"
#include "rules.hpp"

#include <optional>
#include <utility>

class Engine {
public:
    // starts a new game at tick zero. zero dimensions select random side lengths.
    // returns true on success. generation failure returns false and preserves the current game.
    bool reset(int rows = 0, int cols = 0, std::uint32_t seed = 0) {
        auto next = initial(rows, cols, seed);
        if (!next) return false;
        state = std::move(next);
        return true;
    }

    // loads a valid saved state, preserving its board, tick, idle count and result.
    void load(States saved) {
        state = std::move(saved);
    }

    // resolves one half-turn using actions[0] for red and actions[1] for blue.
    // returns true when time advances, including when both players pass or submit invalid moves.
    // returns false while the engine is empty or after the game ends, keeping the state unchanged.
    bool step(const std::array<Action, 2>& actions) {
        if (!state || state->result != Phases::Ongoing) return false;
        // ::step() calls the rule function that settles both moves, growth and the game result.
        ::step(*state, actions);
        return true;
    }

    // returns the current observation for player 0 or 1, including after the game ends.
    // returns std::nullopt while the engine is empty or when the player index is invalid.
    std::optional<Observation> observe(int player) const {
        if (!state || (player != 0 && player != 1)) return std::nullopt;
        return ::observe(*state, player);
    }

    // returns a copy of the full state for replay and inspection. strategy input comes from observe().
    // returns std::nullopt before reset() or load() provides a state.
    std::optional<States> snapshot() const {
        return state;
    }

private:
    std::optional<States> state;
};
