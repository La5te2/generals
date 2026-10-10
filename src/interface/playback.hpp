// Replay playback control: present a recorded game over time without putting playback state in the window.
// Playback combines a Replay reader with a Clock to manage timed advancement, pause/resume and seeking.
// After changing position, it publishes observations and playback status through the shared MatchSnapshot contract.
// Replay owns file decoding and position reconstruction; Clock supplies elapsed-time deadlines.
// The application calls these operations and handles navigation and automatic restart. The renderer only reads the result.
#pragma once

#include "clock.hpp"
#include "replay.hpp"
#include "session.hpp"

namespace NEBULA {
    class Playback {
    public:
        bool load(const std::filesystem::path& path, int milliseconds, std::string& error) {
            if (!replay.load(path, error)) return false;
            clock.interval = std::chrono::milliseconds(milliseconds);
            clock.reset();
            if (replay.length()) clock.resume();
            publish();
            return true;
        }
        bool loaded() const { return replay.loaded(); }
        std::size_t cursor() const { return replay.cursor(); }
        std::size_t length() const { return replay.length(); }
        const MatchSnapshot& snapshot() const { return published; }
        bool advance() {
            if (!clock.consume()) return false;
            return seek(cursor() + 1);
        }
        bool seek(std::size_t tick) {
            if (!replay.loaded() || !replay.seek(tick)) return false;
            if (cursor() == length()) clock.pause();
            bool running = clock.running();
            clock.reset();
            if (running) clock.resume();
            publish();
            return true;
        }
        void toggle() {
            if (!published.active()) return;
            if (clock.running()) clock.pause();
            else clock.resume();
            published.state = clock.running() ? MatchState::Playing : MatchState::Paused;
        }
        void stop() {
            clock.pause();
            if (loaded()) published.state = MatchState::Finished;
        }
        // Set the duration of one playback half-turn, starting a full interval at the new duration.
        // Preserve playing/paused state and the current position; recorded actions and tick numbers do not change.
        void setInterval(int milliseconds) {
            bool running = clock.running();
            clock.interval = std::chrono::milliseconds(milliseconds);
            clock.reset();
            if (running) clock.resume();
        }
    private:
        Replay replay;
        Clock clock;
        MatchSnapshot published;
        void publish() {
            published.views = std::make_shared<const std::array<Observation, 3>>(matchViews(replay.state()));
            published.names = replay.names();
            published.state = cursor() == length() ? MatchState::Finished :
                              clock.running() ? MatchState::Playing : MatchState::Paused;
        }
    };
}
