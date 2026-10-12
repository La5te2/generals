// Server application entry point; owns process startup and shutdown, while RoomServer owns the room service.
#include "server.hpp"
#include <charconv>
#include <chrono>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {
    volatile std::sig_atomic_t stopping = 0;
    // Signal handlers only set the exit flag; normal stack unwinding performs server cleanup.
    void stop(int) { stopping = 1; }
}
// Validate command-line options, start the listener and wait for SIGINT/SIGTERM; report failures with exit code 1.
// RoomServer destruction stops the listener and joins its workers before this process exits.
// The default 127.0.0.1:42817 accepts local connections only.
// For a machine with a reachable public IP, run: server --bind 0.0.0.0 --port 42817
// This listens on all local IPv4 interfaces; clients use ws://PUBLIC_IP:42817 in CUSTOM's ADDRESS.
// Allow inbound TCP 42817 in both the cloud security group and the OS firewall, preferably only
// from the players' IPs. Direct public access needs no tunnel; 0.0.0.0 is not a client address.
// Equal nonempty Room IDs select a private room; empty IDs enter public matchmaking.
// A Room ID identifies a room, not a password or authentication credential.
int main(int argc, char** argv) try {
    std::string address = "127.0.0.1";
    int port = 42817;
    for (int index = 1; index < argc; ++index) {
        std::string_view option = argv[index];
        if (option == "--help") {
            std::cout << "server [--bind ADDRESS] [--port 1-65535]\nDefault: 127.0.0.1:42817. Stop with Ctrl+C.\n";
            return 0;
        }
        if ((option != "--bind" && option != "--port") || ++index == argc) throw std::runtime_error("Expected --bind ADDRESS or --port 1-65535");
        std::string_view value = argv[index];
        if (option == "--bind") { address = value; if (address.empty()) throw std::runtime_error("Empty bind address"); }
        else {
            auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), port);
            if (error != std::errc{} || end != value.data() + value.size() || port < 1 || port > 65535)
                throw std::runtime_error("Port must be between 1 and 65535");
        }
    }
    std::signal(SIGINT, stop); std::signal(SIGTERM, stop);
    NEBULA::RoomServer server(address, static_cast<std::uint16_t>(port));
    std::cout << "Listening on " << address << ':' << server.port() << " (WebSocket). Press Ctrl+C to stop.\n" << std::flush;
    while (!stopping) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
