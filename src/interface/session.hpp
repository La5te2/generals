// shared session interface: let drawing and human controls work with either local or online play.
// sessions publish display data and accept input here. each session handles its own game updates.
#pragma once

#include "engine/observe.hpp"
#include "engine/actions.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace NEBULA {
    enum class MatchState { Empty, Active, Finished };

    // one published value keeps observations, session status and input queues together for the interface.
    struct MatchSnapshot {
        // local play supplies three perspectives. online play fills only the account's perspective.
        std::shared_ptr<const std::array<Observation, 3>> views = std::make_shared<const std::array<Observation, 3>>();
        MatchState state = MatchState::Empty;
        bool running = false;
        std::string error;
        std::array<std::vector<Action>, 2> queued;
        // local and replay use these labels. online fills server usernames in red/blue order at game_start.
        std::array<std::string, 2> names{"RED", "BLUE"};
        std::string status;
        int player = -1; // online account's seat, assigned when the server starts the game.
    };

    // human input submits or cancels queued moves through the active session.
    class PlayerInput {
    public:
        virtual ~PlayerInput() = default;
        virtual bool enqueue(int player, const Action& action) = 0;
        virtual std::optional<Action> cancel(int player, bool all) = 0;
    };
}
