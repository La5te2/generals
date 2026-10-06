// a C boundary lets Python batch the header-only engine and reuse the deployed strategy's observation memory.
#include "features.hpp"
#include "engine/initial.hpp"
#include "engine/rules.hpp"
#include <random>
#include <string>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
    thread_local std::string failure;
    template<class Operation> int checked(Operation operation) {
        try { operation(); failure.clear(); return 0; }
        catch (const std::exception& error) { failure = error.what(); return -1; }
    }
    struct Game {
        States state{18, 18};
        std::array<Learning::Features, 2> input;
        bool done = true;
        explicit Game(int side) : input{Learning::Features(side), Learning::Features(side)} {}
        void observe() { for (int player = 0; player < 2; ++player) input[player].update(::observe(state, player)); }
    };
    struct Arena {
        std::vector<Game> games;
        std::mt19937 random;
        int side, minDistance = 3, maxDistance = 4, horizon = 2048;
        Arena(int count, int size, std::uint32_t seed) : random(seed), side(size) {
            games.reserve(count);
            for (int game = 0; game < count; ++game) games.emplace_back(size);
        }
        int pick(int limit) { return std::uniform_int_distribution<int>(0, limit - 1)(random); }

        States generate() {
            // reuse the local map generator, then choose connected spawn cells at the curriculum's BFS distance.
            // terrain and city guards remain part of the existing mainstream 1v1 environment.
            for (int attempt = 0; attempt < 128; ++attempt) {
                int rows = 17 + pick(std::min(23, side) - 16), cols = 17 + pick(std::min(23, side) - 16);
                auto saved = initial(rows, cols, random());
                if (!saved) continue;
                auto& board = saved->board;
                std::vector<int> empty;
                for (int pos = 0; pos < board.size(); ++pos) {
                    auto& cell = board.at(pos / cols, pos % cols);
                    if (cell.terrain == Terrain::General) cell = {};
                    if (cell.terrain == Terrain::Plain) empty.push_back(pos);
                }
                std::shuffle(empty.begin(), empty.end(), random);
                for (int red : empty) {
                    std::array<int, Capacity> distance, queue{};
                    distance.fill(-1);
                    int head = 0, tail = 0;
                    queue[tail++] = red; distance[red] = 0;
                    std::vector<int> candidates;
                    while (head < tail) {
                        int pos = queue[head++];
                        if (distance[pos] >= minDistance) candidates.push_back(pos);
                        if (distance[pos] == maxDistance) continue;
                        for (int next : InitialDetail::neighbors(board, pos)) {
                            if (next < 0 || distance[next] >= 0 || board.at(next / cols, next % cols).terrain != Terrain::Plain) continue;
                            distance[next] = distance[pos] + 1;
                            queue[tail++] = next;
                        }
                    }
                    if (candidates.empty()) continue;
                    int blue = candidates[pick(static_cast<int>(candidates.size()))];
                    board.at(red / cols, red % cols) = {1, 0, Terrain::General};
                    board.at(blue / cols, blue % cols) = {1, 1, Terrain::General};
                    return std::move(*saved);
                }
            }
            throw std::runtime_error("Map generation exhausted its attempts");
        }
    };
}

EXPORT const char* arena_error() { return failure.c_str(); }
EXPORT void* arena_create(int count, int side, std::uint32_t seed) {
    void* result = nullptr;
    checked([&] {
        if (count < 1 || count > 4096 || side < 18 || side > dim) throw std::invalid_argument("Invalid arena dimensions");
        result = new Arena(count, side, seed);
    });
    return result;
}
EXPORT void arena_destroy(void* handle) { delete static_cast<Arena*>(handle); }
EXPORT int arena_reset(void* handle, int minDistance, int maxDistance, int horizon) {
    return checked([&] {
        if (!handle || minDistance < 3 || maxDistance < minDistance || maxDistance > Capacity || horizon < 1 || horizon > 50000) {
            throw std::invalid_argument("Invalid curriculum settings");
        }
        auto& arena = *static_cast<Arena*>(handle);
        arena.minDistance = minDistance; arena.maxDistance = maxDistance; arena.horizon = horizon;
        for (auto& game : arena.games) {
            if (!game.done) continue;
            game.state = arena.generate();
            for (auto& input : game.input) input.reset();
            game.observe();
            game.done = false;
        }
    });
}
EXPORT int arena_read(void* handle, float* spatial, std::uint8_t* legal, float* temporal) {
    return checked([&] {
        if (!handle || !spatial || !legal || !temporal) throw std::invalid_argument("Invalid observation buffers");
        auto& arena = *static_cast<Arena*>(handle);
        for (auto& game : arena.games) for (const auto& input : game.input) {
            std::copy(input.spatial.begin(), input.spatial.end(), spatial); spatial += input.spatial.size();
            std::copy(input.legal.begin(), input.legal.end(), legal); legal += input.legal.size();
            std::copy(input.temporal.begin(), input.temporal.end(), temporal); temporal += input.temporal.size();
        }
    });
}
EXPORT int arena_step(void* handle, const std::int32_t* actions, float* rewards, std::uint8_t* terminal, std::uint8_t* truncated) {
    return checked([&] {
        if (!handle || !actions || !rewards || !terminal || !truncated) throw std::invalid_argument("Invalid step buffers");
        auto& arena = *static_cast<Arena*>(handle);
        for (std::size_t index = 0; index < arena.games.size(); ++index) {
            auto& game = arena.games[index];
            if (game.done) throw std::logic_error("Reset completed games before stepping");
            std::array<Action, 2> moves{Learning::decode(actions[2 * index], arena.side), Learning::decode(actions[2 * index + 1], arena.side)};
            ::step(game.state, moves);
            terminal[index] = game.state.result != Phases::Ongoing;
            truncated[index] = !terminal[index] && game.state.tick >= static_cast<std::uint64_t>(arena.horizon);
            game.done = terminal[index] || truncated[index];
            rewards[2 * index] = game.state.result == Phases::RedWin ? 1.0f : game.state.result == Phases::BlueWin ? -1.0f : 0.0f;
            rewards[2 * index + 1] = -rewards[2 * index];
            game.observe();
        }
    });
}

// inference uses the same encoder with observations decoded from the public process protocol.
EXPORT void* encoder_create(int side) {
    void* result = nullptr;
    checked([&] {
        if (side < 2 || side > dim) throw std::invalid_argument("Invalid encoder size");
        result = new Learning::Features(side);
    });
    return result;
}
EXPORT void encoder_destroy(void* handle) { delete static_cast<Learning::Features*>(handle); }
EXPORT int encoder_update(void* handle, int rows, int cols, int player, const std::int64_t* values,
    float* spatial, std::uint8_t* legal, float* temporal) {
    return checked([&] {
        if (!handle || !values || !spatial || !legal || !temporal || rows < 1 || cols < 1 || rows > dim || cols > dim || player < 0 || player > 1) {
            throw std::invalid_argument("Invalid encoder input");
        }
        Observation view;
        view.rows = rows; view.cols = cols; view.player = player;
        view.tick = static_cast<std::uint64_t>(values[0]);
        view.land[player] = static_cast<int>(values[1]); view.armies[player] = values[2];
        view.land[1 - player] = static_cast<int>(values[3]); view.armies[1 - player] = values[4];
        int area = rows * cols;
        for (int pos = 0; pos < area; ++pos) {
            auto terrain = values[5 + pos], owner = values[5 + area + pos], army = values[5 + 2 * area + pos];
            if (terrain < 0 || terrain > 5 || owner < 0 || owner > 2 || army < 0) throw std::invalid_argument("Invalid observation cell");
            view.cells[pos] = {static_cast<ViewTerrain>(terrain), static_cast<std::int8_t>(owner == 1 ? player : owner == 2 ? 1 - player : -1), army};
        }
        auto& input = *static_cast<Learning::Features*>(handle);
        input.update(view);
        std::copy(input.spatial.begin(), input.spatial.end(), spatial);
        std::copy(input.legal.begin(), input.legal.end(), legal);
        std::copy(input.temporal.begin(), input.temporal.end(), temporal);
    });
}
