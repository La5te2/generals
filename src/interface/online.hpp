// online session: exchange account observations and actions with a server, keeping its rules authoritative.
#pragma once

#include "session.hpp"

namespace NEBULA {
    enum class Server { Bot, Main };

    struct OnlineConfig {
        Server server = Server::Bot;
        std::string username, userId, command;
        std::string room; // empty selects ranked 1v1. otherwise join this private room.
        std::string proxy; // optional HTTP CONNECT proxy, for example http://127.0.0.1:10090.
    };

    class OnlineMatch : public PlayerInput {
    public:
        OnlineMatch();
        ~OnlineMatch();
        OnlineMatch(const OnlineMatch&) = delete;
        OnlineMatch& operator=(const OnlineMatch&) = delete;

        // validate locally, then connect on a worker. each start uses a fresh connection and agent process.
        bool start(const OnlineConfig& config);
        // cancel matchmaking or leave the game, stop the agent and join the worker. the last snapshot remains readable.
        void stop();
        // copy the latest account-visible board and status under the mailbox lock, without querying the server.
        MatchSnapshot snapshot() const;
        // queue a human move for the assigned seat. ownership and army legality are checked by the server when executed.
        bool enqueue(int player, const Action& action) override;
        // undo the newest pending move or clear the route. return the first removed move for cursor placement.
        std::optional<Action> cancel(int player, bool all) override;

    private:
        struct Session;
        std::unique_ptr<Session> session;
    };
}
