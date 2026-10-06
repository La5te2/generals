// route-based strategy: remember revealed terrain, compare complete routes and execute one planned step at a time.
#pragma once

#include "engine/actions.hpp"
#include "engine/observe.hpp"
#include <algorithm>
#include <deque>
#include <queue>
#include <vector>

class Simple {
public:
    Action act(const Observation& observation) {
        view = &observation;
        if (observation.tick < lastTick || rows != view->rows || cols != view->cols) reset();
        rows = view->rows; cols = view->cols; lastTick = view->tick;
        for (int pos = 0; pos < rows * cols; ++pos) {
            const auto& cell = view->cells[pos];
            if (visible(cell)) {
                memory[pos] = cell;
                seen[pos] = true;
                if (cell.terrain == ViewTerrain::General && cell.owner == view->player) general = pos;
            } else if (!seen[pos]) memory[pos] = cell;
        }
        if (general < 0) return {};

        // defense interrupts a route only when a visible enemy can reach the general soon.
        auto home = distances(general);
        int threat = -1;
        for (int pos = 0; pos < rows * cols; ++pos) {
            const auto& cell = view->cells[pos];
            if (cell.owner != 1 - view->player || home[pos] <= 0 || home[pos] > 8) continue;
            if (cell.army - home[pos] <= view->cells[general].army + home[pos] / 2) continue;
            if (threat < 0 || home[pos] < home[threat]) threat = pos;
        }
        if (threat >= 0) {
            auto rescue = defend(threat, home);
            if (rescue.type == ActionType::Move) { route.clear(); return rescue; }
        }

        // a known general is a concrete objective. a route must arrive with enough troops after every fight.
        Plan attack;
        for (int pos = 0; pos < rows * cols; ++pos) {
            if (memory[pos].terrain != ViewTerrain::General || memory[pos].owner != 1 - view->player) continue;
            attack = toward(pos, true);
            if (!attack.path.empty()) { route.clear(); return move(attack.path[0], attack.path[1]); }
        }

        if (route.size() >= 2) {
            int from = route[0], to = route[1];
            if (canAdvance(from, to) && safeDeparture(from, to)) {
                Action action = move(from, to);
                route.pop_front();
                return action;
            }
            route.clear();
        }

        Plan best;
        for (int pos = 0; pos < rows * cols; ++pos) {
            if (memory[pos].terrain != ViewTerrain::City || view->cells[pos].owner == view->player) continue;
            auto city = toward(pos, false);
            if (city.score > best.score) best = std::move(city);
        }
        auto sources = armies();
        for (int source : sources) {
            auto plan = expand(source);
            if (plan.score > best.score) best = std::move(plan);
        }
        if (best.path.size() < 2) return {};
        // one-troop excursions repeatedly pay the same transit cost. wait for a useful expedition from a producer.
        int source = best.path.front();
        if (view->cells[source].army < 4 && producer(source) && threat < 0 && view->tick % 50 < 44) return {};
        route.assign(best.path.begin(), best.path.end());
        Action action = move(route[0], route[1]);
        route.pop_front();
        return action;
    }

private:
    struct Plan {
        std::vector<int> path;
        double score = 0;
        std::int64_t army = 0;
        double gain = 0;
    };
    const Observation* view = nullptr;
    std::array<ViewCell, Capacity> memory{};
    std::array<bool, Capacity> seen{};
    std::deque<int> route;
    int rows = 0, cols = 0, general = -1;
    std::uint64_t lastTick = 0;

    void reset() { memory = {}; seen = {}; route.clear(); general = -1; }
    static bool visible(const ViewCell& cell) {
        return cell.terrain != ViewTerrain::Fog && cell.terrain != ViewTerrain::Obstacle;
    }
    std::array<int, 4> neighbors(int pos) const {
        int row = pos / cols, col = pos % cols;
        return {row > 0 ? pos - cols : -1, row + 1 < rows ? pos + cols : -1,
                col > 0 ? pos - 1 : -1, col + 1 < cols ? pos + 1 : -1};
    }
    bool open(int pos) const {
        return pos >= 0 && memory[pos].terrain != ViewTerrain::Mountain && memory[pos].terrain != ViewTerrain::Obstacle;
    }
    bool producer(int pos) const {
        return memory[pos].terrain == ViewTerrain::General || memory[pos].terrain == ViewTerrain::City;
    }
    std::array<int, Capacity> distances(int target) const {
        std::array<int, Capacity> distance;
        distance.fill(-1);
        std::queue<int> pending;
        pending.push(target); distance[target] = 0;
        while (!pending.empty()) {
            int pos = pending.front(); pending.pop();
            for (int next : neighbors(pos)) {
                if (!open(next) || distance[next] >= 0) continue;
                distance[next] = distance[pos] + 1;
                pending.push(next);
            }
        }
        return distance;
    }
    std::vector<int> armies() const {
        std::vector<int> sources;
        for (int pos = 0; pos < rows * cols; ++pos) {
            if (view->cells[pos].owner == view->player && view->cells[pos].army > 1) sources.push_back(pos);
        }
        std::stable_sort(sources.begin(), sources.end(), [&](int left, int right) {
            return view->cells[left].army > view->cells[right].army;
        });
        if (sources.size() > 8) sources.resize(8);
        return sources;
    }
    Action move(int from, int to, bool half = false) const {
        return {ActionType::Move, from / cols, from % cols,
            to == from - cols ? Direction::Up : to == from + cols ? Direction::Down :
            to < from ? Direction::Left : Direction::Right, half};
    }
    bool safeDeparture(int source, int target = -1) const {
        if (source != general) return true;
        for (int next : neighbors(source)) {
            if (next < 0 || view->cells[next].owner != 1 - view->player || view->cells[next].army <= 2) continue;
            // capturing the attacking stack protects the general. another adjacent stack still prevents departure.
            if (next == target && view->cells[source].army - 1 > view->cells[next].army) continue;
            return false;
        }
        return true;
    }
    bool canAdvance(int from, int to) const {
        if (view->cells[from].owner != view->player || view->cells[from].army < 2 || !open(to)) return false;
        const auto& cell = view->cells[to];
        return visible(cell) && (cell.owner == view->player || view->cells[from].army - 1 > cell.army);
    }
    // carry the surviving stack through a route, charging one garrison per move and each defender's army.
    bool extend(Plan& plan, int next) const {
        const auto& cell = visible(view->cells[next]) ? view->cells[next] : memory[next];
        auto defense = cell.army;
        bool own = view->cells[next].owner == view->player;
        if (!own && cell.owner >= 0 && producer(next)) defense += static_cast<std::int64_t>(plan.path.size() / 2);
        if (plan.army <= 1 || (!own && plan.army - 1 <= defense)) return false;
        plan.army = plan.army - 1 + (own ? cell.army : -defense);
        if (!own) {
            // compare income over the next 50 half-turns. enemy land also denies its next land-growth income.
            plan.gain += 1 + (cell.owner == 1 - view->player ? 1 : 0);
            if (producer(next)) plan.gain += 25;
            if (!seen[next]) plan.gain += 0.25;
        }
        plan.path.push_back(next);
        plan.score = plan.gain / static_cast<double>(plan.path.size() - 1);
        return true;
    }
    Plan toward(int target, bool winning) const {
        auto distance = distances(target);
        Plan best;
        for (int source : armies()) {
            if (distance[source] <= 0) continue;
            Plan plan{{source}, 0, view->cells[source].army, 0};
            int pos = source;
            while (pos != target) {
                int choice = -1;
                std::int64_t largest = -1;
                for (int next : neighbors(pos)) {
                    if (next < 0 || distance[next] != distance[pos] - 1) continue;
                    if (pos == source && !safeDeparture(source, next)) continue;
                    auto candidate = plan;
                    if (extend(candidate, next) && candidate.army > largest) { choice = next; largest = candidate.army; }
                }
                if (choice < 0) break;
                extend(plan, choice); pos = choice;
            }
            if (pos == target && (best.path.empty() || (winning ? plan.path.size() < best.path.size() : plan.score > best.score))) best = std::move(plan);
        }
        return best;
    }
    Action defend(int enemy, const std::array<int, Capacity>& home) const {
        int deadline = home[enemy];
        auto target = toward(enemy, true);
        if (target.path.size() >= 2 && target.path.size() - 1 <= static_cast<std::size_t>(deadline)) {
            return move(target.path[0], target.path[1]);
        }
        // reinforce along friendly territory only, so a defensive route cannot spend its troops on expansion.
        int bestSource = -1, bestNext = -1;
        double bestRate = 0;
        for (int source : armies()) {
            if (source == general || home[source] <= 0 || home[source] > deadline) continue;
            for (int next : neighbors(source)) {
                if (next < 0 || home[next] != home[source] - 1 || view->cells[next].owner != view->player) continue;
                double rate = static_cast<double>(view->cells[source].army - 1) / home[source];
                if (rate > bestRate) { bestRate = rate; bestSource = source; bestNext = next; }
            }
        }
        return bestSource >= 0 ? move(bestSource, bestNext) : Action{};
    }
    Plan expand(int source) const {
        std::vector<Plan> beam{{{source}, 0, view->cells[source].army, 0}};
        Plan best;
        // bounded beam search rewards newly taken cells, while friendly transit only consumes time.
        for (int depth = 0; depth < 12 && !beam.empty(); ++depth) {
            std::vector<Plan> nextBeam;
            for (const auto& plan : beam) for (int next : neighbors(plan.path.back())) {
                if (!open(next) || std::find(plan.path.begin(), plan.path.end(), next) != plan.path.end()) continue;
                if (plan.path.size() == 1 && !safeDeparture(source, next)) continue;
                auto candidate = plan;
                if (!extend(candidate, next)) continue;
                if (candidate.score > best.score || (candidate.score == best.score && candidate.gain > best.gain)) best = candidate;
                nextBeam.push_back(std::move(candidate));
            }
            std::stable_sort(nextBeam.begin(), nextBeam.end(), [](const Plan& left, const Plan& right) {
                return left.gain + 0.02 * left.army > right.gain + 0.02 * right.army;
            });
            if (nextBeam.size() > 24) nextBeam.resize(24);
            beam = std::move(nextBeam);
        }
        return best;
    }
};
