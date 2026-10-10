// shared match data and input contracts keep rendering and human controls independent of each mode's implementation.
// MatchState and MatchSnapshot describe match progress, observations, players and queued actions for display, including replays.
// PlayerInput lets human controls submit or cancel moves in local, online and LAN matches through the same interface.
// game updates, connections, worker threads and snapshot synchronization belong to the individual mode implementations.
#pragma once

#include "engine/observe.hpp"
#include "engine/actions.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace NEBULA {
    enum class MatchState {
        Empty,      // No session has been started.
        Connecting, // Establishing a peer or server connection.
        Waiting,    // Connected or listening, but no match has started.
        Playing,    // The match or replay is advancing.
        Paused,     // An unfinished local match or replay is paused.
        Finishing,  // Play has ended; final updates or recording transfer are still pending.
        Finished    // Session work has ended. Its last snapshot remains available.
    };

    // one published value keeps observations, session status and input queues together for the interface.
    struct MatchSnapshot {
        // local play supplies three perspectives. online play fills only the account's perspective.
        std::shared_ptr<const std::array<Observation, 3>> views = std::make_shared<const std::array<Observation, 3>>();
        MatchState state = MatchState::Empty;
        bool active() const { return state != MatchState::Empty && state != MatchState::Finished; }
        bool running() const { return state == MatchState::Playing; }
        bool acceptsInput() const { return running() || state == MatchState::Paused; }
        std::string error;
        std::array<std::vector<Action>, 2> queued;
        // Local uses default labels; replay restores recorded names; online and LAN fill participant usernames.
        std::array<std::string, 2> names{"RED", "BLUE"};
        std::string status;
        int player = -1; // This window's online or LAN seat; -1 until assigned.
        int host = -1; // LAN participant accepting the peer connection and running the engine.
    };

    // Both local games and playback expose these three perspectives through the same snapshot contract.
    inline std::array<Observation, 3> matchViews(const States& state) {
        std::array<Observation, 3> next{observe(state, 0), observe(state, 1)};
        next[2] = next[0];
        auto& observation = next[2];
        for (int row = 0; row < observation.rows; ++row) {
            for (int col = 0; col < observation.cols; ++col) {
                const Cell& cell = state.board.at(row, col);
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
        return next;
    }

    // human input submits or cancels queued moves through the active session.
    class PlayerInput {
    public:
        virtual ~PlayerInput() = default;
        virtual bool enqueue(int player, const Action& action) = 0;
        virtual std::optional<Action> cancel(int player, bool all) = 0;
    };
}
