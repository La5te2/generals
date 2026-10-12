// Self-hosted WebSocket service: accept players, pair public/private rooms and own authoritative games.
// Each room advances a LocalMatch independently; clients receive only their view until the game ends.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace NEBULA {
    class RoomServer {
    public:
        // Start listening and matchmaking immediately. Port 0 asks the OS to choose a free port.
        // Listener startup errors propagate as exceptions to the caller.
        explicit RoomServer(const std::string& address = "127.0.0.1", std::uint16_t port = 42817);
        // Stop admission, notify connected players and wait for lobby and room workers to exit.
        ~RoomServer();
        RoomServer(const RoomServer&) = delete;
        RoomServer& operator=(const RoomServer&) = delete;
        // Return the actual listening port, including one assigned by the OS.
        std::uint16_t port() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
}
