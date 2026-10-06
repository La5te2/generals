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
        void stop();
        MatchSnapshot snapshot() const;
        bool enqueue(int player, const Action& action) override;
        std::optional<Action> cancel(int player, bool all) override;

    private:
        struct Session;
        std::unique_ptr<Session> session;
    };
}
