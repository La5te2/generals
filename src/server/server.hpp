// Self-hosted WebSocket service: accept players, pair public/private rooms and own authoritative games.
// Each room advances a LocalMatch independently; clients receive only their view until the game ends.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace NEBULA {
    class RoomServer {
    public:
        explicit RoomServer(const std::string& address = "127.0.0.1", std::uint16_t port = 42817);
        ~RoomServer();
        RoomServer(const RoomServer&) = delete;
        RoomServer& operator=(const RoomServer&) = delete;
        std::uint16_t port() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
}
