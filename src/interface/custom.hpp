// client-side session for custom-server matches, responsible for network state and its presentation to the interface.
// connection and cleanup run off the window thread. this client never listens or discovers other players.
#pragma once

#include "session.hpp"
#include <filesystem>

namespace NEBULA {
    struct ClientConfig {
        std::string address, username, room, command; // address is a ws:// or wss:// server URL.
        std::filesystem::path directory;
        std::string proxy; // optional HTTP CONNECT proxy with no fallback to a direct connection.
    };

    class ClientMatch : public PlayerInput {
    public:
        ClientMatch();
        // request departure and wait for the worker. use leave() earlier when the caller must remain responsive.
        ~ClientMatch();
        ClientMatch(const ClientMatch&) = delete;
        ClientMatch& operator=(const ClientMatch&) = delete;
        // validate configuration and retry any pending save before launching a fresh connection worker.
        // true means startup was accepted, not that matching succeeded. inspect snapshot() for subsequent progress/errors.
        bool start(const ClientConfig& config);
        // request departure asynchronously: cancel waiting or surrender, allowing the worker to receive the final replay.
        void stop();
        // the same request for navigation/exit. opened() remains true until the worker finishes, including transfer/cleanup.
        void leave();
        // true from worker startup through connection, matching and final transfer, not merely while the socket is open.
        bool opened() const;
        // copy the last published board, queues and status without waiting for network work.
        MatchSnapshot snapshot() const;
        // after the worker ends, retry a failed replay write. keep the bytes on failure. an empty destination cannot discard them.
        bool savePending(const std::filesystem::path& directory, std::string& error);
        // queue a human move for this client's seat. check bounds now and let the server check execution-time legality.
        bool enqueue(int player, const Action& action) override;
        // request undo/clear and update the displayed route immediately. return the first removed move for cursor placement.
        std::optional<Action> cancel(int player, bool all) override;
    private:
        struct Session;
        std::unique_ptr<Session> session;
    };
}
