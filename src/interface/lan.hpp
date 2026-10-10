// LAN play: discover peers on a selected local network and run one direct two-player room per window.
// leaving either side closes the room; a new session discovers and matches peers again.
#pragma once

#include "session.hpp"
#include <filesystem>

namespace NEBULA {
    struct LanConfig {
        std::string address, username, room, command; // address selects a local IPv4 adapter, not a remote endpoint.
        std::filesystem::path directory;
        std::string proxy; // HTTP CONNECT for outgoing game connections; discovery stays on the selected LAN.
    };

    class LanMatch : public PlayerInput {
    public:
        LanMatch();
        ~LanMatch();
        LanMatch(const LanMatch&) = delete;
        LanMatch& operator=(const LanMatch&) = delete;
        bool start(const LanConfig& config);
        void stop(); // surrender and close the room, retaining the final snapshot.
        void leave(); // asynchronous; opened() stays true until replay transfer and cleanup finish.
        bool opened() const;
        MatchSnapshot snapshot() const;
        bool savePending(const std::filesystem::path& directory, std::string& error);
        bool enqueue(int player, const Action& action) override;
        std::optional<Action> cancel(int player, bool all) override;

    private:
        struct Session;
        std::unique_ptr<Session> session;
    };
}
