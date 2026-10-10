// Discover available peers on local IPv4 networks; game traffic uses a separate direct connection.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace NEBULA {
    struct Peer {
        std::string id, address, room, host;
        std::uint16_t port = 0;
        bool available = false;
        std::chrono::steady_clock::time_point seen;
    };

    class Discovery {
    public:
        // Empty selects all up, multicast-capable IPv4 interfaces. Otherwise bind discovery to this local IP.
        Discovery(const std::string& local, std::string id, std::string room, std::uint16_t port);
        ~Discovery();
        Discovery(const Discovery&) = delete;
        Discovery& operator=(const Discovery&) = delete;
        void poll(bool available, const std::string& host);
        const std::vector<Peer>& peers() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };
}
