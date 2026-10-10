// Shared validation for anonymous HTTP CONNECT proxies used by WebSocket clients.
#pragma once

#include <rtc/configuration.hpp>
#include <charconv>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace NEBULA {
    inline std::optional<rtc::ProxyServer> httpProxy(std::string_view address) {
        if (address.empty()) return std::nullopt;
        try {
            if (!address.starts_with("http://") || address.size() > 2048 ||
                address.find_first_of("/?#@ \t\r\n", 7) != std::string_view::npos)
                throw std::invalid_argument("Invalid proxy URL");
            auto colon = address.rfind(':');
            int port = 0;
            if (colon == std::string_view::npos || colon <= 6 || colon + 1 == address.size())
                throw std::invalid_argument("Missing proxy port");
            auto [end, error] = std::from_chars(address.data() + colon + 1, address.data() + address.size(), port);
            if (error != std::errc{} || end != address.data() + address.size() || port < 1 || port > 65535)
                throw std::invalid_argument("Invalid proxy port");
            rtc::ProxyServer result{std::string(address)};
            if (result.type != rtc::ProxyServer::Type::Http || result.hostname.empty() || !result.port || result.username || result.password)
                throw std::invalid_argument("Unsupported proxy");
            return result;
        } catch (const std::exception&) {
            throw std::invalid_argument("Proxy requires http://host:port without authentication");
        }
    }
}
