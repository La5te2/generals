// replay playback controller that owns presentation time and the displayed replay snapshot, independently of the window.
// Replay owns decoding and position reconstruction. the application owns navigation/restart and the renderer only draws.
#pragma once

#include "clock.hpp"
#include "replay.hpp"
#include "session.hpp"

namespace NEBULA {
    class Playback {
    public:
        // load at tick zero and begin timed playback. on load failure preserve the previous reader, clock and snapshot.
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
        // advance at most one half-turn when its clock interval expires. return whether the displayed position changed.
        bool advance() {
            if (!clock.consume()) return false;
            return seek(cursor() + 1);
        }
        // reconstruct an absolute half-turn, publish all perspectives and restart the interval if playing. the end pauses it.
        bool seek(std::size_t tick) {
            if (!replay.loaded() || !replay.seek(tick)) return false;
            if (cursor() == length()) clock.pause();
            bool running = clock.running();
            clock.reset();
            if (running) clock.resume();
            publish();
            return true;
        }
        // pause/resume an unfinished replay without changing its position or discarding the remaining interval.
        void toggle() {
            if (!published.active()) return;
            if (clock.running()) clock.pause();
            else clock.resume();
            published.state = clock.running() ? MatchState::Playing : MatchState::Paused;
        }
        // end this playback session at its current position for navigation/exit rather than a replay toolbar Stop.
        void stop() {
            clock.pause();
            if (loaded()) published.state = MatchState::Finished;
        }
        // set the duration of one playback half-turn, starting a full interval at the new duration.
        // preserve playing/paused state and the current position. recorded actions and tick numbers do not change.
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
        // rebuild the immutable display views and derive playback state from the cursor and clock.
        void publish() {
            published.views = std::make_shared<const std::array<Observation, 3>>(matchViews(replay.state()));
            published.names = replay.names();
            published.state = cursor() == length() ? MatchState::Finished : clock.running() ? MatchState::Playing : MatchState::Paused;
        }
    };
}
