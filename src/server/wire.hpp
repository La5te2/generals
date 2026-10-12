// Custom-server wire format and bounded WebSocket inboxes shared by the client and room service.
// Callbacks only collect packets; the owning session/room thread validates and handles them in order.
#pragma once

#include "engine/protocol.hpp"
#include <rtc/websocket.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace NEBULA::Wire {
    // A session sends join, waits for paired, then sends ready. The server starts after both players are ready.
    // state contains one player's observation, its pending queue and the last processed input id (ack).
    // After a terminal state, ordered replay chunks carry .gior bytes as hex; recorded confirms receipt, then closed ends the session.
    // leave cancels matchmaking or surrenders an ongoing game. ping/pong keep idle queues and paused startup detectable.
    using Json = nlohmann::json;
    using Socket = std::shared_ptr<rtc::WebSocket>;
    using Time = std::chrono::steady_clock;
    inline constexpr std::size_t packetLimit = 256 * 1024, recordingLimit = 16 * 1024 * 1024, chunkLimit = 32768;
    inline constexpr int version = 1, halfTurnMilliseconds = 500;

    inline void require(bool value, const char* error) { if (!value) throw std::runtime_error(error); }
    inline bool nameValid(const std::string& value) {
        return !value.empty() && value.size() <= 64 &&
            std::all_of(value.begin(), value.end(), [](unsigned char c) { return c >= 32 && c <= 126; });
    }
    inline std::string encode(const Action& action) {
        std::ostringstream stream;
        require(Protocol::writeAction(stream, action), "Invalid action");
        return stream.str();
    }
    inline Action decode(const Json& text) {
        require(text.is_string() && text.get_ref<const std::string&>().size() <= 100, "Invalid action");
        std::istringstream stream(text.get<std::string>());
        auto action = Protocol::readAction(stream);
        stream >> std::ws;
        require(action.has_value() && stream.eof(), "Invalid action");
        return *action;
    }
    inline Json parse(const std::string& text) {
        auto data = Json::parse(text, [](int depth, Json::parse_event_t, Json&) {
            require(depth <= 12, "Packet nesting is too deep"); return true;
        });
        require(data.is_object(), "Expected a packet object");
        return data;
    }
    inline std::string hex(std::string_view bytes) {
        std::string result;
        result.reserve(bytes.size() * 2);
        for (unsigned char byte : bytes) { result += "0123456789abcdef"[byte >> 4]; result += "0123456789abcdef"[byte & 15]; }
        return result;
    }
    inline void append(std::string& bytes, const Json& message) {
        auto offset = message.at("offset").get<std::size_t>();
        const auto& text = message.at("data").get_ref<const std::string&>();
        require(offset == bytes.size() && text.size() <= chunkLimit * 2 && text.size() % 2 == 0 &&
                text.size() / 2 <= recordingLimit - bytes.size(), "Invalid recording chunk");
        for (std::size_t i = 0; i < text.size(); i += 2) {
            auto digit = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
            int a = digit(text[i]), b = digit(text[i + 1]);
            require(a >= 0 && b >= 0, "Invalid recording bytes");
            bytes += static_cast<char>((a << 4) | b);
        }
    }

    struct Inbox {
        std::mutex mutex;
        std::deque<std::string> packets;
        std::size_t bytes = 0;
        std::string failure;
        bool closed = false;
    };
    struct Connection {
        Socket socket;
        std::shared_ptr<Inbox> inbox = std::make_shared<Inbox>();
        bool closed = false; // Snapshot taken with take(), after all preceding message callbacks.
        Time::time_point last = Time::now(), heartbeat = last, arrived = last;
        explicit Connection(Socket source) : socket(std::move(source)) {
            std::weak_ptr<rtc::WebSocket> weak = socket;
            socket->onMessage([mail = inbox, weak](rtc::message_variant message) {
                auto text = std::get_if<std::string>(&message);
                bool rejected;
                {
                    std::lock_guard lock(mail->mutex);
                    rejected = !text || text->size() > packetLimit || mail->packets.size() >= 128 ||
                               (text && mail->bytes + text->size() > 2 * 1024 * 1024);
                    if (rejected) mail->failure = "Invalid packet or receive queue overflow";
                    else if (mail->failure.empty()) { mail->bytes += text->size(); mail->packets.push_back(std::move(*text)); }
                }
                if (rejected) if (auto source = weak.lock()) source->forceClose();
            });
            socket->onError([mail = inbox](std::string error) {
                std::lock_guard lock(mail->mutex);
                if (mail->failure.empty()) mail->failure = "WebSocket: " + error.substr(0, 256);
            });
            socket->onClosed([mail = inbox] { std::lock_guard lock(mail->mutex); mail->closed = true; });
        }
        ~Connection() { socket->close(); }
        std::deque<std::string> take() {
            std::lock_guard lock(inbox->mutex);
            // Deliver a queued protocol error/closed packet before observing transport shutdown.
            if (!inbox->failure.empty() && inbox->packets.empty()) throw std::runtime_error(inbox->failure);
            closed = inbox->closed;
            std::deque<std::string> packets;
            packets.swap(inbox->packets); inbox->bytes = 0;
            return packets;
        }
        void send(const Json& data) {
            require(socket->isOpen(), "Connection closed");
            require(socket->bufferedAmount() < 2 * 1024 * 1024, "Peer is not receiving updates");
            socket->send(data.dump());
        }
        bool control(const Json& message) {
            auto type = message.at("type").get<std::string>();
            last = Time::now();
            if (type == "ping") { send({{"type", "pong"}}); return true; }
            return type == "pong";
        }
        void pulse() {
            auto now = Time::now();
            require(now - last < std::chrono::seconds(30), "Connection timed out");
            if (now - heartbeat >= std::chrono::seconds(5)) { send({{"type", "ping"}}); heartbeat = now; }
        }
        void reject(const std::string& error) noexcept {
            try { send({{"type", "error"}, {"message", error.substr(0, 256)}}); socket->close(); }
            catch (...) { socket->forceClose(); }
        }
    };
}
