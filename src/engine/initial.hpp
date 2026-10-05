#pragma once

#include "states.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

// default 1v1 generation, adapted from the official v31.4.3 client:
// https://generals.io/generals-main-prod-v31.4.3-485dbf55.js.
namespace InitialDetail {
    inline double sample(std::mt19937& random) {
        return static_cast<double>(random()) / 4294967296.0;
    }

    inline int pick(std::mt19937& random, int count) {
        return static_cast<int>(sample(random) * count);
    }

    inline int distance(const Board& board, int first, int second) {
        return std::abs(first / board.cols() - second / board.cols()) +
               std::abs(first % board.cols() - second % board.cols());
    }

    // negative entries represent directions that leave the board.
    inline std::array<int, 4> neighbors(const Board& board, int pos) {
        int row = pos / board.cols(), col = pos % board.cols();
        return {row > 0 ? pos - board.cols() : -1,
                row + 1 < board.rows() ? pos + board.cols() : -1,
                col > 0 ? pos - 1 : -1,
                col + 1 < board.cols() ? pos + 1 : -1};
    }

    inline Terrain terrain(const Board& board, int pos) {
        return board.at(pos / board.cols(), pos % board.cols()).terrain;
    }

    using Routes = std::array<std::array<int, 4>, Capacity>;

    // shuffle each cell's neighbors so graph searches follow a seed-dependent order.
    inline Routes routes(const Board& board, std::mt19937& random) {
        Routes result{};
        for (int pos = 0; pos < board.size(); ++pos) {
            result[pos].fill(-1);
            int count = 0;
            for (int next : neighbors(board, pos)) {
                if (next >= 0) result[pos][count++] = next;
            }
            while (count > 0) {
                int choice = pick(random, count);
                std::swap(result[pos][--count], result[pos][choice]);
            }
        }
        return result;
    }

    // weight the number of open cells at each walking distance, from one through fourteen.
    // the nearest layers matter most, and the first layer contributes at most three cells.
    inline double spawnScore(const Board& board, const Routes& paths, int start) {
        std::array<int, Capacity> depths{};
        depths.fill(-1);
        std::array<int, Capacity> queue{};
        std::array<int, 15> counts{};
        int head = 0, tail = 0;
        queue[tail++] = start;
        depths[start] = 0;
        while (head < tail) {
            int pos = queue[head++], depth = depths[pos];
            ++counts[depth];
            if (depth == 14) continue;
            for (int next : paths[pos]) {
                if (next < 0 || depths[next] >= 0 || terrain(board, next) != Terrain::Plain) continue;
                depths[next] = depth + 1;
                queue[tail++] = next;
            }
        }
        counts[1] = std::min(3, counts[1]);
        double score = 0;
        double weight = static_cast<int>(std::pow(1.18, 15));
        for (int depth = 1; depth < 15; ++depth) {
            score += weight * counts[depth];
            weight /= depth < 7 ? 1.18 : 1.1;
            if (depth < 5) weight /= 1.6;
        }
        return score;
    }

    // resample distant spawn pairs until their expansion scores meet the current tolerance.
    // each rejected score comparison relaxes the fairness setting by 0.005.
    inline std::optional<std::array<int, 2>> spawns(const Board& board, const Routes& paths, std::mt19937& random) {
        std::array<bool, Capacity> valid{};
        bool available = false;
        for (int pos = 0; pos < board.size(); ++pos) {
            if (terrain(board, pos) != Terrain::Plain) continue;
            for (int next : paths[pos]) {
                if (next >= 0 && terrain(board, next) == Terrain::Plain) valid[pos] = true;
            }
            available |= valid[pos];
        }
        if (!available) return std::nullopt;
        std::array<int, 2> generals{-1, -1};
        std::vector<int> pending{-1};
        int rejected = 0;
        for (int attempt = 0; attempt < 1000; ++attempt) {
            // the retry list stores old positions. a new spawn matching a later entry is sampled again.
            for (int previous : pending) {
                for (int player = 0; player < 2; ++player) {
                    if (generals[player] != previous) continue;
                    do { generals[player] = pick(random, board.size()); } while (!valid[generals[player]]);
                }
            }
            if (distance(board, generals[0], generals[1]) <= 9 * (1.8 - 0.1 * 2)) {
                pending = {generals[0], generals[1]};
                continue;
            }
            double fairness = 0.5 - 0.005 * rejected;
            double ratio = 1 - std::pow(1 - fairness, 4) - 0.02;
            if (ratio <= 0) return generals;
            double red = spawnScore(board, paths, generals[0]);
            double blue = spawnScore(board, paths, generals[1]);
            if (red / std::max(1.0, blue) < ratio) {
                pending = sample(random) > 0.25 ? std::vector<int>{generals[0], generals[1]} : std::vector<int>{generals[1], generals[0]};
            } else if (blue / std::max(1.0, red) < ratio) {
                sample(random);
                pending = {generals[1], generals[0]};
            } else return generals;
            ++rejected;
        }
        return std::nullopt;
    }

    struct CityGroup {
        int id;
        std::vector<int> cells;
        bool neutral = false; // a distant or equidistant city placed on its own.
    };

    struct CityInfo {
        int difference; // distance from red minus distance from blue.
        int total; // sum of the two distances.
        bool neutral; // equally distant, or distant from both players.
    };

    // selected mountains become tentative cities while a plan is being evaluated.
    // distance maps allow travel through those cities, modelling routes after their capture.
    class CityPlacement {
    public:
        CityPlacement(const Board& map, const Routes& paths, const std::array<int, 2>& spawn, std::mt19937& rng)
            : board(map), adjacency(paths), generals(spawn), random(rng) {
            positions.fill(-1);
            for (int pos = 0; pos < board.size(); ++pos) {
                if (terrain(board, pos) == Terrain::Mountain) restore(pos);
            }
            refresh();
        }

        // default city fairness is 0.5. its distance-deficit tolerance is 3.375.
        // a neutral city is equally distant, or at least seventeen steps from each general.
        std::optional<std::vector<CityGroup>> plan(int desired) {
            int attempts = std::max(2 * static_cast<int>(remaining.size()), 8 * desired);
            while (attempts-- > 0 && !remaining.empty()) {
                if (count() >= desired && std::abs(deficit) <= tolerance) break;
                std::vector<int> candidates = remaining;
                bool placed = false;
                while (!candidates.empty()) {
                    int choice = pick(random, static_cast<int>(candidates.size()));
                    int pos = candidates[choice];
                    candidates[choice] = candidates.back();
                    candidates.pop_back();
                    if (!take(pos)) continue;
                    int id = nextId++;
                    groups.push_back({id, {pos}, false});
                    selected[pos] = true;
                    refresh();
                    prune();
                    auto city = info(pos);
                    if (city) {
                        groups[groupIndex(id)].neutral = city->neutral;
                        score();
                        if (city->neutral) {
                            placed = true;
                            break;
                        }
                        auto matches = partners(*city);
                        if (!matches.empty()) {
                            int partner = matches[pick(random, static_cast<int>(matches.size()))];
                            if (take(partner)) {
                                groups[groupIndex(id)].cells.push_back(partner);
                                selected[partner] = true;
                                refresh();
                                prune();
                                if (repair() && std::abs(deficit) <= tolerance) {
                                    placed = true;
                                    break;
                                }
                            }
                        }
                    }
                    remove(id);
                }
                if (!placed) break;
            }
            if (count() == 0 || std::abs(deficit) > tolerance) return std::nullopt;
            return groups;
        }

    private:
        const Board& board;
        const Routes& adjacency;
        const std::array<int, 2>& generals;
        std::mt19937& random;
        std::vector<CityGroup> groups;
        std::vector<int> remaining;
        std::array<int, Capacity> positions{};
        std::array<bool, Capacity> selected{};
        std::array<std::array<int, Capacity>, 2> distances{};
        int nextId = 0;
        double deficit = 0;
        static constexpr double tolerance = 3.375;

        int groupIndex(int id) const {
            for (int index = 0; index < static_cast<int>(groups.size()); ++index) {
                if (groups[index].id == id) return index;
            }
            return -1;
        }

        int count() const {
            int result = 0;
            for (const auto& group : groups) result += static_cast<int>(group.cells.size());
            return result;
        }

        // moving the last candidate into the removed slot determines the next random selection's order.
        bool take(int pos) {
            int index = positions[pos];
            if (index < 0) return false;
            int last = remaining.back();
            remaining[index] = last;
            positions[last] = index;
            remaining.pop_back();
            positions[pos] = -1;
            return true;
        }

        void restore(int pos) {
            if (positions[pos] >= 0) return;
            positions[pos] = static_cast<int>(remaining.size());
            remaining.push_back(pos);
        }

        std::optional<CityInfo> info(int pos) const {
            int red = distances[0][pos], blue = distances[1][pos];
            if (red > board.size() || blue > board.size()) return std::nullopt;
            return CityInfo{red - blue, red + blue, red == blue || (red >= 17 && blue >= 17)};
        }

        // the pair deficit is the absolute difference between the two total distances.
        static int pairDeficit(const CityInfo& first, const CityInfo& second) {
            return first.difference <= second.difference ? first.total - second.total : second.total - first.total;
        }

        void score() {
            deficit = 0;
            for (const auto& group : groups) {
                if (group.neutral) continue;
                std::vector<CityInfo> cities;
                for (int pos : group.cells) {
                    auto city = info(pos);
                    if (city) {
                        deficit += city->difference * 1.5;
                        cities.push_back(*city);
                    }
                }
                if (cities.size() == 2) deficit += pairDeficit(cities[0], cities[1]);
            }
        }

        // a mountain receives a distance but expansion stops there, unless it is a selected city.
        // recompute distances after tentative cities open or close routes.
        void refresh() {
            for (int player = 0; player < 2; ++player) {
                auto& depth = distances[player];
                depth.fill(board.size() + 1);
                std::array<int, Capacity> queue{};
                int head = 0, tail = 0;
                queue[tail++] = generals[player];
                depth[generals[player]] = 0;
                while (head < tail) {
                    int pos = queue[head++];
                    for (int next : adjacency[pos]) {
                        if (next < 0 || depth[next] <= depth[pos] + 1) continue;
                        depth[next] = depth[pos] + 1;
                        if (terrain(board, next) != Terrain::Mountain || selected[next]) queue[tail++] = next;
                    }
                }
            }
            score();
        }

        // opening a city may change the shortest paths and make a formerly neutral city favor one side.
        // those cities return to the candidate pool before the next pairing decision.
        void prune() {
            bool changed = false;
            for (int index = static_cast<int>(groups.size()) - 1; index >= 0; --index) {
                const auto& group = groups[index];
                if (!group.neutral) continue;
                int pos = group.cells[0];
                auto city = info(pos);
                if (city && !city->neutral) {
                    selected[pos] = false;
                    restore(pos);
                    groups.erase(groups.begin() + index);
                    changed = true;
                }
            }
            if (changed) refresh();
        }

        void remove(int id) {
            int index = groupIndex(id);
            if (index < 0) return;
            for (int pos : groups[index].cells) {
                selected[pos] = false;
                restore(pos);
            }
            groups.erase(groups.begin() + index);
            refresh();
        }

        // a paired city favors the opposite player, while both the partial and final deficits stay small.
        std::vector<int> partners(const CityInfo& first) const {
            std::vector<int> matches;
            for (int pos : remaining) {
                auto city = info(pos);
                if (!city || city->neutral) continue;
                if ((first.difference < 0 && city->difference <= 0) ||
                    (first.difference > 0 && city->difference >= 0)) continue;
                double partial = deficit + city->difference * 1.5;
                int pair = pairDeficit(first, *city);
                if (std::abs(partial) >= tolerance || std::abs(pair) >= tolerance ||
                    std::abs(partial + pair) >= tolerance) continue;
                matches.push_back(pos);
            }
            return matches;
        }

        // if changed routes break an existing pair, keep the neutral member and pair its partner again.
        bool repair() {
            int attempts = std::max({4 * static_cast<int>(groups.size()), 2 * static_cast<int>(remaining.size()), 1});
            bool changed = true;
            while (changed && attempts-- > 0) {
                changed = false;
                for (int index = 0; index < static_cast<int>(groups.size()); ++index) {
                    if (groups[index].neutral || groups[index].cells.size() != 2) continue;
                    auto first = info(groups[index].cells[0]), second = info(groups[index].cells[1]);
                    if (!first || !second) continue;
                    int invalid = -1;
                    if (first->neutral) invalid = 0;
                    else if (second->neutral || (first->difference < 0) == (second->difference < 0)) invalid = 1;
                    if (invalid < 0) continue;
                    int id = groups[index].id;
                    int neutral = groups[index].cells[invalid], retained = groups[index].cells[1 - invalid];
                    groups.push_back({nextId++, {neutral}, true});
                    groups[index].cells = {retained};
                    score();
                    auto city = info(retained);
                    if (!city || city->neutral) return false;
                    auto matches = partners(*city);
                    if (matches.empty()) return false;
                    int partner = matches[pick(random, static_cast<int>(matches.size()))];
                    take(partner);
                    groups[groupIndex(id)].cells.push_back(partner);
                    selected[partner] = true;
                    refresh();
                    prune();
                    if (std::abs(deficit) > tolerance) return false;
                    changed = true;
                    break;
                }
            }
            return !changed;
        }
    };

    // the generals need a route through plain land.
    // each general needs at least one city within six Manhattan steps.
    inline bool valid(const Board& board, const std::array<int, 2>& generals) {
        std::array<int, 2> nearby{};
        int cities = 0;
        for (int pos = 0; pos < board.size(); ++pos) {
            if (terrain(board, pos) != Terrain::City) continue;
            ++cities;
            for (int player = 0; player < 2; ++player) {
                if (distance(board, pos, generals[player]) <= 6) ++nearby[player];
            }
        }
        if (cities >= 2 && (nearby[0] == 0 || nearby[1] == 0)) return false;
        std::array<bool, Capacity> seen{};
        std::array<int, Capacity> queue{};
        int head = 0, tail = 0;
        queue[tail++] = generals[0];
        seen[generals[0]] = true;
        while (head < tail) {
            int pos = queue[head++];
            if (pos == generals[1]) return true;
            for (int next : neighbors(board, pos)) {
                if (next < 0 || seen[next]) continue;
                if (terrain(board, next) != Terrain::Plain && next != generals[1]) continue;
                seen[next] = true;
                queue[tail++] = next;
            }
        }
        return false;
    }
}

// each random side length is round(18 + 5 * r * r), where r is uniform in [0, 1).
// this gives lengths from 18 to 23, with smaller sizes appearing more often.
// rows == 0 randomizes the height, and cols == 0 randomizes the width. positive values fix that length.
// return std::nullopt for an invalid size or after all 100000 map attempts fail.
inline std::optional<States> initial(int rows, int cols, std::uint32_t seed) {
    if (rows < 0 || rows > dim || cols < 0 || cols > dim) return std::nullopt;
    if (rows > 0 && cols > 0 && rows + cols - 2 < 15) return std::nullopt;
    std::mt19937 random(seed);
    for (int attempt = 0; attempt < 100000; ++attempt) {
        int width = static_cast<int>(std::round(18 + 5 * std::pow(InitialDetail::sample(random), 2)));
        int height = static_cast<int>(std::round(18 + 5 * std::pow(InitialDetail::sample(random), 2)));
        if (rows > 0) height = rows;
        if (cols > 0) width = cols;
        States state(height, width);
        auto paths = InitialDetail::routes(state.board, random);
        double cities = 9 + 2 * InitialDetail::sample(random);
        double mountains = state.board.size() * (0.2 + 0.08 * InitialDetail::sample(random)) + cities;
        // sample mountain positions with replacement. repeated picks keep the existing mountain.
        for (int index = 0; index < mountains; ++index) {
            int pos = InitialDetail::pick(random, state.board.size());
            state.board.at(pos / width, pos % width).terrain = Terrain::Mountain;
        }
        auto generals = InitialDetail::spawns(state.board, paths, random);
        if (!generals) continue;
        for (int player = 0; player < 2; ++player) {
            int pos = (*generals)[player];
            state.board.at(pos / width, pos % width) = {1, static_cast<std::int8_t>(player), Terrain::General};
        }
        // retry city plans on the same terrain up to eight times before generating another map.
        for (int plan = 0; plan < 8; ++plan) {
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    Cell& cell = state.board.at(row, col);
                    if (cell.terrain == Terrain::City) cell = {0, -1, Terrain::Mountain};
                }
            }
            InitialDetail::CityPlacement placement(state.board, paths, *generals, random);
            auto groups = placement.plan(static_cast<int>(std::round(cities)));
            if (!groups) continue;
            // each pair shares one guard value. round() gives 40 and 50 half the probability of interior values.
            for (const auto& group : *groups) {
                int army = static_cast<int>(std::round(40 + 10 * InitialDetail::sample(random)));
                for (int pos : group.cells) state.board.at(pos / width, pos % width) = {army, -1, Terrain::City};
            }
            if (InitialDetail::valid(state.board, *generals)) return state;
        }
    }
    return std::nullopt;
}
