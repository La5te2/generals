// replay storage and playback: share .gior files between local recordings and downloaded games.
#pragma once

#include "engine/engine.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace NEBULA {
    struct ReplayMove {
        std::uint64_t tick;
        int player;
        Action action;
    };

    struct Recording {
        States initial, previous;
        std::vector<ReplayMove> moves;
        std::optional<int> surrendered;

        explicit Recording(const States& state) : initial(state), previous(state) {}
        // store moves against the position where they were submitted. empty half-turns are implicit.
        void append(const States& state, const std::array<Action, 2>& actions);
    };

    // validate or create the destination before a game starts. actual write errors are reported when saving.
    bool replayDirectory(const std::filesystem::path& directory, std::string& error);
    std::optional<std::filesystem::path> saveReplay(const Recording& record, const std::filesystem::path& directory,
                                                   std::string& error);

    class Replay {
    public:
        // validate and reconstruct a candidate before replacing the currently loaded replay.
        bool load(const std::filesystem::path& path, std::string& error);
        bool seek(std::size_t halfTurn);
        const States& state() const { return *position; }
        const std::array<std::string, 2>& names() const { return players; }
        bool loaded() const { return position.has_value(); }
        std::size_t cursor() const { return position ? static_cast<std::size_t>(position->tick) : 0; }
        std::size_t length() const { return turns.size(); }

    private:
        struct Turn {
            std::array<Action, 2> actions{};
            std::array<bool, 2> surrendered{};
        };
        // sparse checkpoints bound backward-seek work while keeping complete boards out of the file.
        static constexpr std::size_t checkpointInterval = 128;
        std::vector<Turn> turns;
        std::vector<States> checkpoints;
        std::optional<States> position;
        std::array<std::string, 2> players{"RED", "BLUE"};

        void advance(States& state) const;
    };
}
