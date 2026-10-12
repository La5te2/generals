// Custom-server client: submit human/agent input, publish the assigned view and receive completed replays.
// Connection and cleanup run off the window thread. This client never listens or discovers other players.
#pragma once

#include "session.hpp"
#include <filesystem>

namespace NEBULA {
    struct ClientConfig {
        std::string address, username, room, command; // address is a ws:// or wss:// server URL.
        std::filesystem::path directory;
        std::string proxy; // optional HTTP CONNECT proxy; never falls back to a direct connection.
    };

    class ClientMatch : public PlayerInput {
    public:
        ClientMatch();
        ~ClientMatch();
        ClientMatch(const ClientMatch&) = delete;
        ClientMatch& operator=(const ClientMatch&) = delete;
        bool start(const ClientConfig& config);
        void stop(); // surrender, receive the final recording and keep its final board on screen.
        void leave(); // asynchronous; opened() remains true until transfer and cleanup finish.
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
