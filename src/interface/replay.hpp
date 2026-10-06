// local replays: record snapshots and submitted actions, and encode their differences for compact file storage.
#pragma once

#include "engine/engine.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace NEBULA {
    struct ReplayFrame {
        std::string board;
        std::array<Action, 2> actions{};
        std::uint64_t tick = 0;
        std::uint32_t idle = 0;
        Phases result = Phases::Ongoing;
    };

    struct Recording {
        int rows, cols;
        std::vector<ReplayFrame> frames;

        explicit Recording(const States& initial);
        // record the position after both submitted actions have been settled, including scheduled growth.
        void append(const States& state, const std::array<Action, 2>& actions);
    };

    // validate or create the destination before a game starts. actual write errors are reported when saving.
    bool replayDirectory(const std::filesystem::path& directory, std::string& error);
    std::optional<std::filesystem::path> saveReplay(const Recording& record, const std::filesystem::path& directory,
                                                   std::string& error);

    class Replay {
    public:
        // replace the current replay only after the whole file has passed validation.
        bool load(const std::filesystem::path& path, std::string& error);
        bool seek(std::size_t halfTurn);
        const States& state() const { return *position; }
        bool loaded() const { return position.has_value(); }
        std::size_t cursor() const { return current; }
        std::size_t length() const { return frames.empty() ? 0 : frames.size() - 1; }

    private:
        std::vector<ReplayFrame> frames;
        std::optional<States> position;
        std::size_t current = 0;
    };
}
