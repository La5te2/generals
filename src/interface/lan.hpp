// LAN play: join a player-hosted WebSocket lobby and its public or private two-player rooms.
// the room creator controls room lifetime; a locally opened lobby survives returning to the menu.
#pragma once

#include "session.hpp"
#include <filesystem>

namespace NEBULA {
    struct LanConfig {
        std::string address, username, room, command;
        int milliseconds = 500;
        std::filesystem::path directory;
        std::string proxy; // outgoing HTTP CONNECT connection; empty permits direct connection and local hosting.
    };

    class LanMatch : public PlayerInput {
    public:
        LanMatch();
        ~LanMatch();
        LanMatch(const LanMatch&) = delete;
        LanMatch& operator=(const LanMatch&) = delete;
        bool start(const LanConfig& config);
        void stop(); // surrender this match while keeping the room connection.
        void restart(const std::filesystem::path& directory = {}); // both players must be ready for the next match.
        void leave();
        bool opened() const;
        MatchSnapshot snapshot() const;
        bool savePending(const std::filesystem::path& directory, std::string& error);
        bool enqueue(int player, const Action& action) override;
        std::optional<Action> cancel(int player, bool all) override;

    private:
        struct Lobby;
        struct Session;
        std::vector<std::unique_ptr<Lobby>> lobbies;
        std::unique_ptr<Session> session;
    };
}
