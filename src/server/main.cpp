// Headless entry point for the self-hosted room service. Tunnel its listening TCP port to a public endpoint.
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
    void stop(int) { stopping = 1; }
}
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
