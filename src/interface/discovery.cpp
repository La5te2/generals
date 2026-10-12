// LAN discovery advertises each window independently; there is no directory server or traffic relay.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <mswsock.h>
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
#include <cstring>
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
        struct Adapter {
            in_addr address{}, broadcast{};
            unsigned int index = 0;
            bool multicast = false;
        };
        [[maybe_unused]] in_addr broadcastAddress(in_addr address, unsigned int prefix) {
            if (prefix == 0 || prefix >= 31) return {};
            in_addr result{}; result.s_addr = htonl(ntohl(address.s_addr) | (0xffffffffu >> prefix));
            return result;
        }
        std::vector<Adapter> interfaces(const std::string& selected) {
            std::vector<Adapter> result;
            auto add = [&](const sockaddr* address, in_addr broadcast, unsigned int index, bool multicast) {
                if (!address || address->sa_family != AF_INET) return;
                auto value = reinterpret_cast<const sockaddr_in*>(address)->sin_addr;
                char text[INET_ADDRSTRLEN]{};
                inet_ntop(AF_INET, &value, text, sizeof(text));
                if (!selected.empty() && selected != text) return;
                if (std::none_of(result.begin(), result.end(), [&](const Adapter& other) { return other.address.s_addr == value.s_addr; }))
                    result.push_back({value, broadcast, index, multicast});
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
                if (adapter->OperStatus != IfOperStatusUp) continue;
                for (auto* entry = adapter->FirstUnicastAddress; entry; entry = entry->Next) {
                    if (!entry->Address.lpSockaddr || entry->Address.lpSockaddr->sa_family != AF_INET ||
                        (entry->DadState != IpDadStatePreferred && entry->DadState != IpDadStateDeprecated)) continue;
                    auto address = reinterpret_cast<const sockaddr_in*>(entry->Address.lpSockaddr)->sin_addr;
                    bool loopback = adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK;
                    auto broadcast = loopback ? in_addr{} : broadcastAddress(address, entry->OnLinkPrefixLength);
                    add(entry->Address.lpSockaddr, broadcast, adapter->IfIndex, !(adapter->Flags & IP_ADAPTER_NO_MULTICAST));
                }
            }
#else
            ifaddrs* adapters = nullptr;
            if (getifaddrs(&adapters)) throw std::runtime_error("Cannot enumerate local network adapters");
            for (auto* entry = adapters; entry; entry = entry->ifa_next) {
                if (!(entry->ifa_flags & IFF_UP)) continue;
                in_addr broadcast{};
                if ((entry->ifa_flags & IFF_BROADCAST) && entry->ifa_broadaddr && entry->ifa_broadaddr->sa_family == AF_INET)
                    broadcast = reinterpret_cast<const sockaddr_in*>(entry->ifa_broadaddr)->sin_addr;
                add(entry->ifa_addr, broadcast, if_nametoindex(entry->ifa_name), entry->ifa_flags & (IFF_MULTICAST | IFF_LOOPBACK));
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
        std::vector<Adapter> adapters;
#ifdef _WIN32
        LPFN_WSARECVMSG receiveMessage = nullptr;
#endif
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
                if (!option(socket, SO_BROADCAST, reuse, SOL_SOCKET) || !option(socket, IP_PKTINFO, reuse))
                    throw std::runtime_error("Cannot configure LAN discovery interfaces");
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
                std::erase_if(adapters, [&](Adapter& adapter) {
                    ip_mreq membership{target.sin_addr, adapter.address};
                    adapter.multicast = adapter.multicast && option(socket, IP_ADD_MEMBERSHIP, membership);
                    return !adapter.multicast && !adapter.broadcast.s_addr;
                });
                if (adapters.empty()) throw std::runtime_error("No LAN adapter supports multicast or broadcast discovery");
#ifdef _WIN32
                DWORD ttl = 1, loop = 1;
#else
                unsigned char ttl = 1, loop = 1;
#endif
                if (!option(socket, IP_MULTICAST_TTL, ttl) || !option(socket, IP_MULTICAST_LOOP, loop))
                    throw std::runtime_error("Cannot configure LAN multicast scope");
#ifdef _WIN32
                GUID extension = WSAID_WSARECVMSG;
                DWORD bytes = 0;
                if (WSAIoctl(socket, SIO_GET_EXTENSION_FUNCTION_POINTER, &extension, sizeof(extension),
                             &receiveMessage, sizeof(receiveMessage), &bytes, nullptr, nullptr))
                    throw std::runtime_error("Cannot inspect LAN discovery packet interfaces");
                u_long nonblocking = 1;
                if (ioctlsocket(socket, FIONBIO, &nonblocking)) throw std::runtime_error("Cannot configure LAN discovery socket");
#else
                if (fcntl(socket, F_SETFL, O_NONBLOCK) < 0) throw std::runtime_error("Cannot configure LAN discovery socket");
#endif
            } catch (...) { closeSocket(socket); socket = invalid; throw; }
        }
        ~Impl() { if (socket != invalid) closeSocket(socket); }

        bool send(const std::string& text, const sockaddr_in& destination, const Adapter& adapter) {
            // Explicit source/interface selection keeps discovery on the requested LAN, even with overlapping routes.
#ifdef _WIN32
            WSABUF buffer{static_cast<ULONG>(text.size()), const_cast<char*>(text.data())};
            alignas(WSACMSGHDR) char control[WSA_CMSG_SPACE(sizeof(IN_PKTINFO))]{};
            WSAMSG message{};
            message.name = reinterpret_cast<sockaddr*>(const_cast<sockaddr_in*>(&destination)); message.namelen = sizeof(destination);
            message.lpBuffers = &buffer; message.dwBufferCount = 1; message.Control = {sizeof(control), control};
            auto* header = WSA_CMSG_FIRSTHDR(&message);
            header->cmsg_level = IPPROTO_IP; header->cmsg_type = IP_PKTINFO; header->cmsg_len = WSA_CMSG_LEN(sizeof(IN_PKTINFO));
            IN_PKTINFO info{}; info.ipi_addr = adapter.address; info.ipi_ifindex = adapter.index;
            std::memcpy(WSA_CMSG_DATA(header), &info, sizeof(info));
            DWORD size = 0;
            return WSASendMsg(socket, &message, 0, &size, nullptr, nullptr) == 0 && size == text.size();
#else
            iovec buffer{const_cast<char*>(text.data()), text.size()};
            alignas(cmsghdr) char control[CMSG_SPACE(sizeof(in_pktinfo))]{};
            msghdr message{};
            message.msg_name = const_cast<sockaddr_in*>(&destination); message.msg_namelen = sizeof(destination);
            message.msg_iov = &buffer; message.msg_iovlen = 1; message.msg_control = control; message.msg_controllen = sizeof(control);
            auto* header = CMSG_FIRSTHDR(&message);
            header->cmsg_level = IPPROTO_IP; header->cmsg_type = IP_PKTINFO; header->cmsg_len = CMSG_LEN(sizeof(in_pktinfo));
            in_pktinfo info{}; info.ipi_spec_dst = adapter.address; info.ipi_ifindex = static_cast<int>(adapter.index);
            std::memcpy(CMSG_DATA(header), &info, sizeof(info));
            return sendmsg(socket, &message, 0) == static_cast<ssize_t>(text.size());
#endif
        }
        int receive(std::array<char, 2048>& bytes, sockaddr_in& source, unsigned int& index) {
#ifdef _WIN32
            WSABUF buffer{static_cast<ULONG>(bytes.size()), bytes.data()};
            alignas(WSACMSGHDR) char control[WSA_CMSG_SPACE(sizeof(IN_PKTINFO))]{};
            WSAMSG message{};
            message.name = reinterpret_cast<sockaddr*>(&source); message.namelen = sizeof(source);
            message.lpBuffers = &buffer; message.dwBufferCount = 1; message.Control = {sizeof(control), control};
            DWORD size = 0;
            if (receiveMessage(socket, &message, &size, nullptr, nullptr)) return WSAGetLastError() == WSAEMSGSIZE ? 0 : -1;
            if (message.dwFlags & (MSG_TRUNC | MSG_CTRUNC)) return 0;
            for (auto* header = WSA_CMSG_FIRSTHDR(&message); header; header = WSA_CMSG_NXTHDR(&message, header)) {
                if (header->cmsg_level != IPPROTO_IP || header->cmsg_type != IP_PKTINFO || header->cmsg_len < WSA_CMSG_LEN(sizeof(IN_PKTINFO))) continue;
                IN_PKTINFO info{}; std::memcpy(&info, WSA_CMSG_DATA(header), sizeof(info)); index = info.ipi_ifindex;
            }
#else
            iovec buffer{bytes.data(), bytes.size()};
            alignas(cmsghdr) char control[CMSG_SPACE(sizeof(in_pktinfo))]{};
            msghdr message{};
            message.msg_name = &source; message.msg_namelen = sizeof(source);
            message.msg_iov = &buffer; message.msg_iovlen = 1; message.msg_control = control; message.msg_controllen = sizeof(control);
            auto size = recvmsg(socket, &message, 0);
            if (size < 0) return -1;
            if (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) return 0;
            for (auto* header = CMSG_FIRSTHDR(&message); header; header = CMSG_NXTHDR(&message, header)) {
                if (header->cmsg_level != IPPROTO_IP || header->cmsg_type != IP_PKTINFO || header->cmsg_len < CMSG_LEN(sizeof(in_pktinfo))) continue;
                in_pktinfo info{}; std::memcpy(&info, CMSG_DATA(header), sizeof(info)); index = static_cast<unsigned int>(info.ipi_ifindex);
            }
#endif
            return static_cast<int>(size);
        }
        void poll(bool available, const std::string& host) {
            auto now = Time::now();
            if (now - sent >= 500ms || available != previousAvailable || host != previousHost) {
                auto text = nlohmann::json{{"type", "generals-lan"}, {"id", id}, {"room", room}, {"port", listenPort},
                                           {"available", available}, {"host", host}}.dump();
                bool sentAny = false;
                for (const auto& adapter : adapters) {
                    if (adapter.multicast) sentAny |= send(text, target, adapter);
                    if (adapter.broadcast.s_addr) {
                        auto broadcast = target; broadcast.sin_addr = adapter.broadcast;
                        sentAny |= send(text, broadcast, adapter);
                    }
                }
                if (!sentAny) throw std::runtime_error("Cannot send LAN discovery messages");
                sent = now; previousAvailable = available; previousHost = host;
            }
            // Limit work per poll so unrelated datagrams cannot starve game updates.
            for (int count = 0; count < 64; ++count) {
                std::array<char, 2048> bytes{};
                sockaddr_in source{};
                unsigned int index = 0;
                auto size = receive(bytes, source, index);
                if (size < 0) break;
                if (!size || std::none_of(adapters.begin(), adapters.end(), [&](const Adapter& adapter) { return adapter.index == index; })) continue;
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
