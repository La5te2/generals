// Engine-backed match lifecycle and timing, shared by local play and authoritative server rooms.
#pragma once

#include "clock.hpp"
#include "engine/engine.hpp"
#include "process.hpp"
#include "replay.hpp"
#include "session.hpp"
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace NEBULA {
    struct LocalSnapshot : MatchSnapshot {
        std::filesystem::path saved;
        bool unsaved = false;
        std::shared_ptr<const Recording> completed; // optional finished recording for server transfer, never sent during play.
    };

    class LocalMatch : public PlayerInput {
    public:
        ~LocalMatch();
        // Initialize a seeded game and its agents, then start the half-turn worker; refuse replacement of an active match.
        // An empty command leaves that seat's input to the caller. A destination enables disk recording;
        // retainRecording also keeps the completed game in the snapshot without requiring a destination.
        bool start(const std::array<std::string, 2>& commands, std::uint32_t seed, int milliseconds = 500,
                   const std::filesystem::path& directory = {}, bool retainRecording = false);
        // Finish and join the worker. A supplied seat forfeits; otherwise the sole caller-controlled seat forfeits,
        // or army/land totals decide the winner if both seats have the same input type. An existing result is preserved.
        void stop(int surrender = -1);
        // Retry a finished game's failed write at a new destination without starting another match.
        // Success releases the pending recording; failure retains it and supplies error.
        bool savePending(const std::filesystem::path& destination, std::string& error);
        // Suspend half-turn settlement while preserving queued input and the unused clock interval.
        void pause();
        // Continue an unfinished paused match; this does not restart a finished game.
        void resume();
        // begin a full interval with the new duration, retaining the current running or paused state.
        bool setInterval(int milliseconds);
        // manually settle one half-turn while paused, using replies available at the time of this call.
        bool advance();
        // queued moves execute in order, one per half-turn, with legality checked at execution time.
        bool enqueue(int player, const Action& action) override;
        // cancel the last queued move, or the entire queue. return the first removed move for cursor placement.
        std::optional<Action> cancel(int player, bool all) override;
        // read the latest published board and status, independently of rule updates and agent communication.
        LocalSnapshot snapshot() const;

    private:
        void run();
        bool settle(Clock::Time cutoff);
        void request();
        void finish(int surrender = -1);
        void updateViews();
        void publish();
        bool saveRecording();

        mutable std::mutex mutex;
        std::condition_variable changed;
        std::thread worker;
        Engine engine;
        Clock clock;
        enum class Phase { Empty, Active, Finished };
        Phase phase = Phase::Empty;
        std::array<std::unique_ptr<StrategyProcess>, 2> strategies;
        struct Pending { Action action; Clock::Time received; };
        std::array<std::deque<Pending>, 2> inputs;
        std::string failure, saveError;
        std::optional<Recording> recording;
        std::shared_ptr<const Recording> completed;
        std::filesystem::path directory, saved;
        std::shared_ptr<const std::array<Observation, 3>> views = std::make_shared<const std::array<Observation, 3>>();
        // hold this separate lock only while exchanging the small snapshot, never while computing or stopping processes.
        mutable std::mutex snapshotMutex;
        LocalSnapshot published;
    };
}
