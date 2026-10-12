// replay data: store local recordings and read .gior files from local or downloaded games.
// decode and validate recorded actions, then reconstruct requested positions with the engine and cached checkpoints.
// playback timing and controls belong to playback.hpp. this module has no wall-clock playback state.
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
        std::array<std::string, 2> names{"RED", "BLUE"};

        explicit Recording(const States& state) : initial(state), previous(state) {}
        // called after settlement with its resulting state. record actions legal in previous, then replace previous with state.
        // stored move ticks refer to the position before settlement. passes and empty half-turns remain implicit.
        void append(const States& state, const std::array<Action, 2>& actions);
    };

    // validate or create the destination before a game starts. actual write errors are reported when saving.
    bool replayDirectory(const std::filesystem::path& directory, std::string& error);
    // encode and save a recording. return its path or no value with error on failure.
    std::optional<std::filesystem::path> saveReplay(const Recording& record, const std::filesystem::path& directory,
                                                   std::string& error);
    // produce compressed .gior bytes for disk or network transfer. invalid recording data throws.
    std::string encodeReplay(const Recording& record);
    // write existing bytes via a temporary file. reuse an identical digest-named file and report conflicting contents.
    // this function saves bytes without decoding them. network callers validate the replay before saving.
    std::optional<std::filesystem::path> saveReplayBytes(std::string_view bytes, const std::filesystem::path& directory,
                                                        std::string& error);

    class Replay {
    public:
        // validate and reconstruct a candidate before replacing the currently loaded replay.
        bool load(const std::filesystem::path& path, std::string& error);
        // decode and simulate the recording to validate moves/build checkpoints. success installs it at tick zero.
        // on failure set error and preserve the loaded replay, including its current position.
        bool loadBytes(std::string_view bytes, std::string& error);
        // reconstruct an absolute tick from the current position or a nearby checkpoint. reject out-of-range seeks unchanged.
        bool seek(std::size_t halfTurn);
        // requires loaded(). the reference is valid until this Replay changes or is destroyed.
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
