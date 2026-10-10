// LAN discovery advertises each window independently; there is no directory server or traffic relay.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include "discovery.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <stdexcept>

namespace NEBULA {
    namespace {
        using Time = std::chrono::steady_clock;
        using namespace std::chrono_literals;
        constexpr std::uint16_t port = 42817;
        constexpr const char* group = "239.255.42.17";
#ifdef _WIN32
        using Handle = SOCKET;
        constexpr auto invalid = INVALID_SOCKET;
        void closeSocket(Handle socket) { closesocket(socket); }
        struct Runtime {
            Runtime() { WSADATA data{}; if (WSAStartup(MAKEWORD(2, 2), &data)) throw std::runtime_error("Cannot initialize LAN discovery"); }
            ~Runtime() { WSACleanup(); }
        };
#else
        using Handle = int;
        constexpr auto invalid = -1;
        void closeSocket(Handle socket) { close(socket); }
        struct Runtime {};
#endif
        template<class T> bool option(Handle socket, int name, const T& value, int level = IPPROTO_IP) {
            return setsockopt(socket, level, name, reinterpret_cast<const char*>(&value), sizeof(value)) == 0;
        }
        std::vector<in_addr> interfaces(const std::string& selected) {
            std::vector<in_addr> result;
            auto add = [&](const sockaddr* address) {
                if (!address || address->sa_family != AF_INET) return;
                auto value = reinterpret_cast<const sockaddr_in*>(address)->sin_addr;
                char text[INET_ADDRSTRLEN]{};
                inet_ntop(AF_INET, &value, text, sizeof(text));
                if (!selected.empty() && selected != text) return;
                if (std::none_of(result.begin(), result.end(), [&](in_addr other) { return other.s_addr == value.s_addr; })) result.push_back(value);
            };
#ifdef _WIN32
            ULONG bytes = 16384;
            std::vector<unsigned char> buffer(bytes);
            auto* adapters = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data());
            ULONG status = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_DNS_SERVER, nullptr, adapters, &bytes);
            if (status == ERROR_BUFFER_OVERFLOW) {
                buffer.resize(bytes); adapters = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data());
                status = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_DNS_SERVER, nullptr, adapters, &bytes);
            }
            if (status != NO_ERROR) throw std::runtime_error("Cannot enumerate local network adapters");
            for (auto* adapter = adapters; adapter; adapter = adapter->Next) {
                if (adapter->OperStatus != IfOperStatusUp || (adapter->Flags & IP_ADAPTER_NO_MULTICAST)) continue;
                for (auto* entry = adapter->FirstUnicastAddress; entry; entry = entry->Next) add(entry->Address.lpSockaddr);
            }
#else
            ifaddrs* adapters = nullptr;
            if (getifaddrs(&adapters)) throw std::runtime_error("Cannot enumerate local network adapters");
            for (auto* entry = adapters; entry; entry = entry->ifa_next) {
                if ((entry->ifa_flags & IFF_UP) && (entry->ifa_flags & (IFF_MULTICAST | IFF_LOOPBACK))) add(entry->ifa_addr);
            }
            freeifaddrs(adapters);
#endif
            if (result.empty()) throw std::runtime_error(selected.empty() ? "No usable IPv4 LAN adapter" : "LOCAL IP must belong to an active local adapter");
            return result;
        }
    }

    struct Discovery::Impl {
        Runtime runtime;
        Handle socket = invalid;
        std::vector<in_addr> adapters;
        std::vector<Peer> peers;
        std::string id, room;
        std::uint16_t listenPort;
        sockaddr_in target{};
        Time::time_point sent{};
        bool previousAvailable = false;
        std::string previousHost;

        Impl(const std::string& local, std::string identity, std::string roomId, std::uint16_t listening)
            : adapters(interfaces(local)), id(std::move(identity)), room(std::move(roomId)), listenPort(listening) {
            socket = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
            if (socket == invalid) throw std::runtime_error("Cannot open LAN discovery socket");
            try {
                int reuse = 1;
                if (!option(socket, SO_REUSEADDR, reuse, SOL_SOCKET)) throw std::runtime_error("Cannot share LAN discovery port");
#ifdef SO_REUSEPORT
                option(socket, SO_REUSEPORT, reuse, SOL_SOCKET);
#endif
#ifdef IP_MULTICAST_ALL
                int allInterfaces = 0;
                if (!option(socket, IP_MULTICAST_ALL, allInterfaces)) throw std::runtime_error("Cannot restrict LAN discovery to selected adapters");
#endif
                sockaddr_in binding{};
                binding.sin_family = AF_INET; binding.sin_port = htons(port); binding.sin_addr.s_addr = htonl(INADDR_ANY);
                if (bind(socket, reinterpret_cast<sockaddr*>(&binding), sizeof(binding))) throw std::runtime_error("Cannot bind LAN discovery port 42817");
                target.sin_family = AF_INET; target.sin_port = htons(port);
                inet_pton(AF_INET, group, &target.sin_addr);
                std::erase_if(adapters, [&](in_addr adapter) {
                    ip_mreq membership{target.sin_addr, adapter};
                    return !option(socket, IP_ADD_MEMBERSHIP, membership);
                });
                if (adapters.empty()) throw std::runtime_error("Cannot join LAN discovery group");
#ifdef _WIN32
                DWORD ttl = 1, loop = 1;
#else
                unsigned char ttl = 1, loop = 1;
#endif
                if (!option(socket, IP_MULTICAST_TTL, ttl) || !option(socket, IP_MULTICAST_LOOP, loop))
                    throw std::runtime_error("Cannot configure LAN multicast scope");
#ifdef _WIN32
                u_long nonblocking = 1;
                if (ioctlsocket(socket, FIONBIO, &nonblocking)) throw std::runtime_error("Cannot configure LAN discovery socket");
#else
                if (fcntl(socket, F_SETFL, O_NONBLOCK) < 0) throw std::runtime_error("Cannot configure LAN discovery socket");
#endif
            } catch (...) { closeSocket(socket); socket = invalid; throw; }
        }
        ~Impl() { if (socket != invalid) closeSocket(socket); }

        void poll(bool available, const std::string& host) {
            auto now = Time::now();
            if (now - sent >= 500ms || available != previousAvailable || host != previousHost) {
                auto text = nlohmann::json{{"type", "generals-lan"}, {"id", id}, {"room", room}, {"port", listenPort},
                                           {"available", available}, {"host", host}}.dump();
                bool delivered = false;
                for (auto adapter : adapters) {
                    if (option(socket, IP_MULTICAST_IF, adapter))
                        delivered |= sendto(socket, text.data(), static_cast<int>(text.size()), 0,
                                            reinterpret_cast<sockaddr*>(&target), sizeof(target)) >= 0;
                }
                if (!delivered) throw std::runtime_error("Cannot send LAN discovery messages");
                sent = now; previousAvailable = available; previousHost = host;
            }
            // Limit work per poll so unrelated datagrams cannot starve game updates.
            for (int count = 0; count < 64; ++count) {
                std::array<char, 2048> bytes{};
                sockaddr_in source{};
#ifdef _WIN32
                int length = sizeof(source);
#else
                socklen_t length = sizeof(source);
#endif
                auto size = recvfrom(socket, bytes.data(), static_cast<int>(bytes.size()), 0, reinterpret_cast<sockaddr*>(&source), &length);
                if (size < 0) break;
                try {
                    auto data = nlohmann::json::parse(bytes.data(), bytes.data() + size, nullptr, false);
                    if (!data.is_object() || data.value("type", "") != "generals-lan") continue;
                    Peer peer;
                    peer.id = data.at("id").get<std::string>(); peer.room = data.at("room").get<std::string>();
                    peer.host = data.at("host").get<std::string>();
                    auto validId = [](const std::string& value) {
                        return value.size() == 32 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
                    };
                    if (!validId(peer.id) || peer.id == id || peer.room != room || (!peer.host.empty() && !validId(peer.host))) continue;
                    int remotePort = data.at("port").get<int>();
                    if (remotePort < 1 || remotePort > 65535) continue;
                    peer.port = static_cast<std::uint16_t>(remotePort); peer.available = data.at("available").get<bool>();
                    char address[INET_ADDRSTRLEN]{}; inet_ntop(AF_INET, &source.sin_addr, address, sizeof(address));
                    peer.address = address; peer.seen = now;
                    auto entry = std::find_if(peers.begin(), peers.end(), [&](const Peer& old) { return old.id == peer.id; });
                    if (entry != peers.end()) *entry = std::move(peer);
                    else if (peers.size() < 256) peers.push_back(std::move(peer));
                } catch (const nlohmann::json::exception&) {} // Ignore malformed or unrelated discovery datagrams.
            }
            std::erase_if(peers, [&](const Peer& peer) { return now - peer.seen > 3s; });
        }
    };

    Discovery::Discovery(const std::string& local, std::string id, std::string room, std::uint16_t listenPort)
        : impl(std::make_unique<Impl>(local, std::move(id), std::move(room), listenPort)) {}
    Discovery::~Discovery() = default;
    void Discovery::poll(bool available, const std::string& host) { impl->poll(available, host); }
    const std::vector<Peer>& Discovery::peers() const { return impl->peers; }
}
