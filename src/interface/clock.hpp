#pragma once

#include <chrono>

namespace NEBULA {
    class Clock {
    public:
        using Source = std::chrono::steady_clock;
        using Time = Source::time_point;
        static constexpr auto interval = std::chrono::milliseconds(500);

        bool running() const { return active; }

        // reset() pauses the timer and restores the full 500 ms interval.
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

        // seconds remaining until the timer expires.
        double wait(Time now = Source::now()) const {
            if (!active || now >= deadline) return 0;
            return std::chrono::duration<double>(deadline - now).count();
        }

    private:
        bool active = false;
        Time deadline{};
        Source::duration remaining = interval;
    };
}
