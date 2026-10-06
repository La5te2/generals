#pragma once

#include <chrono>

namespace NEBULA {
    class Clock {
    public:
        using Source = std::chrono::steady_clock;
        using Time = Source::time_point;
        std::chrono::milliseconds interval{500};

        bool running() const { return active; }
        Time expires() const { return deadline; }

        // reset() pauses the timer and restores a full half-turn interval.
        void reset() {
            active = false;
            remaining = interval;
        }

        void resume(Time now = Source::now()) {
            if (active) return;
            deadline = now + remaining;
            active = true;
        }

        // preserve the unused part of this interval while paused.
        void pause(Time now = Source::now()) {
            if (!active) return;
            remaining = deadline > now ? deadline - now : Source::duration::zero();
            active = false;
        }

        // return true when the timer expires, then schedule the next expiration.
        // maintain the original schedule during short delays, and restart the timing after prolonged stalling
        bool consume(Time now = Source::now()) {
            if (!active || now < deadline) return false;
            deadline += interval;
            if (deadline <= now) deadline = now + interval;
            return true;
        }

    private:
        bool active = false;
        Time deadline{};
        Source::duration remaining = interval;
    };
}
