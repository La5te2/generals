// self-hosted WebSocket service: accept players, pair public/private rooms and own authoritative games.
// each room advances a LocalMatch independently. clients receive only their view until the game ends.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace NEBULA {
    class RoomServer {
    public:
        // start listening and matchmaking immediately. port 0 asks the OS to choose a free port.
        // listener startup errors propagate as exceptions to the caller.
        explicit RoomServer(const std::string& address = "127.0.0.1", std::uint16_t port = 42817);
        // stop admission, notify connected players and wait for lobby and room workers to exit.
        ~RoomServer();
        RoomServer(const RoomServer&) = delete;
        RoomServer& operator=(const RoomServer&) = delete;
        // return the actual listening port, including one assigned by the OS.
        std::uint16_t port() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
}
