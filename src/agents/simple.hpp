// route-based agent: remember terrain and enemy movement, compare complete routes and execute one planned step at a time.
#pragma once

#include "engine/actions.hpp"
#include "engine/observe.hpp"
#include <algorithm>
#include <cmath>
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

        home = distances(general);
        assess();
        // defense interrupts a route only when a visible enemy can reach the general soon.
        int threat = -1;
        for (int pos = 0; pos < rows * cols; ++pos) {
            const auto& cell = view->cells[pos];
            if (cell.owner != 1 - view->player || home[pos] <= 0 || home[pos] > 8) continue;
            if (cell.army - home[pos] <= view->cells[general].army + growth(general, home[pos] - 1)) continue;
            if (threat < 0 || home[pos] < home[threat]) threat = pos;
        }
        if (threat >= 0) {
            auto rescue = defend(threat);
            if (rescue.type == ActionType::Move) { route.clear(); return rescue; }
        }

        // a known general is a concrete objective. a route must arrive with enough troops after every fight.
        Plan attack;
        for (int pos = 0; pos < rows * cols; ++pos) {
            if (memory[pos].terrain != ViewTerrain::General || memory[pos].owner != 1 - view->player) continue;
            attack = toward(pos, true);
            if (!attack.path.empty()) { route.clear(); return move(attack.path[0], attack.path[1], attack.half); }
        }

        Plan continuing;
        if (route.size() >= 2) {
            int source = route.front();
            if (view->cells[source].owner == view->player) {
                continuing = {{source}, 0, view->cells[source].army, 0};
                for (std::size_t index = 1; index < route.size(); ++index) {
                    if (!extend(continuing, route[index])) { continuing = {}; break; }
                }
            }
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
        // switching plans must repay one extra move of progress. invalid or unsafe routes receive no preference.
        if (continuing.path.size() >= 2 && continuing.score > 0 &&
            continuing.score >= best.score * (1 - 1.0 / continuing.path.size())) best = std::move(continuing);
        if (best.path.size() < 2 || best.score <= 0 || best.wait > 0) { route.clear(); return {}; }
        route.assign(best.path.begin(), best.path.end());
        Action action = move(route[0], route[1], best.half);
        route.pop_front();
        return action;
    }

private:
    struct Plan {
        std::vector<int> path;
        double score = 0;
        std::int64_t army = 0;
        double gain = 0;
        bool half = false;
        int wait = 0;
        double placement = 0;
    };
    const Observation* view = nullptr;
    Observation previous;
    std::array<ViewCell, Capacity> memory{};
    std::array<bool, Capacity> seen{};
    std::array<double, Capacity> trail{};
    std::array<bool, Capacity> origins{};
    std::array<int, Capacity> home{}, front{};
    std::deque<int> route;
    double alert = 0, advantage = 0;
    std::int64_t guard = 1;
    int contact = -1;
    bool exposed = false;
    int rows = 0, cols = 0, general = -1;
    std::uint64_t lastTick = 0;

    void reset() {
        memory = {}; seen = {}; trail = {}; previous = {}; route.clear(); general = -1;
        origins.fill(true); contact = -1; guard = 1;
        alert = 0; advantage = 0; exposed = false;
    }
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
    std::array<int, Capacity> distances(const std::vector<int>& targets) const {
        std::array<int, Capacity> distance;
        distance.fill(-1);
        std::queue<int> pending;
        for (int target : targets) { pending.push(target); distance[target] = 0; }
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
    std::array<int, Capacity> distances(int target) const { return distances(std::vector<int>{target}); }
    std::int64_t growth(int pos, int ticks) const {
        auto end = view->tick + static_cast<std::uint64_t>(ticks);
        return static_cast<std::int64_t>(end / 50 - view->tick / 50 +
            (producer(pos) ? end / 2 - view->tick / 2 : 0));
    }
    // trail records evidence of enemy territory, including a short guess behind an observed movement.
    // visible empty or friendly cells clear that guess. fading evidence never changes remembered terrain.
    void assess() {
        int own = view->player, enemy = 1 - own;
        auto elapsed = previous.rows > 0 ? view->tick - previous.tick : 0;
        double decay = std::pow(0.95, static_cast<double>(elapsed)), pressure = 0;
        for (int pos = 0; pos < rows * cols; ++pos) {
            const auto& cell = view->cells[pos];
            trail[pos] = visible(cell) ? (cell.owner == enemy ? 1.0 : 0.0) : trail[pos] * decay;
            if (cell.owner != enemy) continue;
            if (contact < 0) {
                contact = pos;
                // an enemy-owned cell has a path from its spawn of at most tick moves.
                // Manhattan distance is a lower bound, including routes through currently hidden obstacles.
                for (int origin = 0; origin < rows * cols; ++origin) {
                    int distance = std::abs(origin / cols - pos / cols) + std::abs(origin % cols - pos % cols);
                    origins[origin] = static_cast<std::uint64_t>(distance) <= view->tick;
                }
            }
            if (std::abs(pos / cols - general / cols) <= 1 && std::abs(pos % cols - general % cols) <= 1) exposed = true;
            if (cell.army > 1 && home[pos] > 0) {
                double force = static_cast<double>(cell.army) / (static_cast<double>(view->cells[general].army) + home[pos]);
                pressure = std::max(pressure, std::min(1.0, force) / (1 + home[pos] / 8.0));
            }
        }
        // matching a source decrease to a neighboring increase distinguishes movement from periodic growth.
        if (previous.rows > 0 && elapsed == 1) {
            auto growth = [&](int pos) { return (view->tick % 50 == 0 ? 1 : 0) + (view->tick % 2 == 0 && producer(pos) ? 1 : 0); };
            for (int pos = 0; pos < rows * cols; ++pos) {
                const auto& before = previous.cells[pos];
                if (view->cells[pos].owner != enemy || !visible(before)) continue;
                auto received = view->cells[pos].army - growth(pos) - (before.owner == enemy ? before.army : -before.army);
                if (received <= 1) continue;
                for (int source : neighbors(pos)) {
                    if (source < 0 || previous.cells[source].owner != enemy || view->cells[source].owner != enemy) continue;
                    auto sent = previous.cells[source].army + growth(source) - view->cells[source].army;
                    if (sent != received) continue;
                    if (home[pos] > 0 && home[source] > home[pos]) pressure = std::max(pressure, 1 / (1 + home[pos] / 8.0));
                    int row = source / cols, col = source % cols;
                    int dr = row - pos / cols, dc = col - pos % cols;
                    for (int reach = 1; reach <= 4; ++reach) {
                        row += dr; col += dc;
                        if (row < 0 || row >= rows || col < 0 || col >= cols) break;
                        int next = row * cols + col;
                        if (!open(next)) break;
                        if (view->cells[next].terrain == ViewTerrain::Fog) trail[next] = std::max(trail[next], 1.0 / (reach + 1));
                        else if (view->cells[next].owner != enemy) break;
                    }
                    break;
                }
            }
        }
        double nextAlert = std::max({pressure, alert * decay, exposed ? 0.25 : 0.0});
        alert = nextAlert;
        // public land and army advantages jointly adjust appetite for enemy contact.
        double ownArmy = static_cast<double>(view->armies[own]), enemyArmy = static_cast<double>(view->armies[enemy]);
        advantage = 0.5 * ((view->land[own] - view->land[enemy]) / static_cast<double>(view->land[own] + view->land[enemy] + 1)
            + (ownArmy - enemyArmy) / (ownArmy + enemyArmy + 1));
        std::vector<int> targets;
        // exposure gives even an unseen attack some weight, using public troops per enemy cell as a modest baseline.
        guard = std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(alert * enemyArmy / std::max(1, view->land[enemy]))));
        for (int pos = 0; pos < rows * cols; ++pos) {
            const auto& cell = view->cells[pos];
            if (seen[pos] && memory[pos].terrain != ViewTerrain::General) origins[pos] = false;
            if (cell.terrain == ViewTerrain::Obstacle || cell.owner == own) origins[pos] = false;
            if (memory[pos].owner == enemy && open(pos)) targets.push_back(pos);
            if (cell.owner == enemy && home[pos] > 0) {
                guard = std::max(guard, cell.army - home[pos] - growth(general, home[pos] - 1));
            }
        }
        front = distances(targets);
        previous = *view;
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
    // check the troops left at home against the earliest visible attack, before that arrival's scheduled growth.
    bool safeDeparture(int source, int target, bool half = false, int wait = 0) const {
        if (source != general) return true;
        auto army = view->cells[source].army + growth(source, wait);
        auto sent = half ? army / 2 : army - 1;
        for (int pos = 0; pos < rows * cols; ++pos) {
            const auto& enemy = view->cells[pos];
            if (enemy.owner != 1 - view->player || home[pos] <= 0 || home[pos] > 8) continue;
            auto available = view->cells[source].army + growth(general, home[pos] - 1);
            if (wait >= home[pos]) {
                if (enemy.army - home[pos] > available) return false;
            } else {
                if (wait == 0 && pos == target && sent > enemy.army) continue;
                if (enemy.army - home[pos] > available - sent && enemy.army - home[pos] <= available) return false;
                // when holding already loses, the remaining army may still seek a counterattack.
                if (home[pos] == 1 && enemy.army - 1 > available - sent) return false;
            }
        }
        return true;
    }
    // support saturates once it covers the observed threat, so extra troops retain their offensive value.
    double position(int pos, std::int64_t army) const {
        double support = home[pos] >= 0 ? alert * static_cast<double>(std::min(army, guard)) / (home[pos] + 1) : 0;
        double pressure = front[pos] >= 0 ? (1 + std::abs(advantage)) * std::sqrt(static_cast<double>(army)) / (front[pos] + 1) : 0;
        return support + pressure;
    }
    // carry the surviving stack through a route, charging one garrison per move and each defender's army.
    bool extend(Plan& plan, int next) const {
        if (!open(next)) return false;
        const auto& cell = visible(view->cells[next]) ? view->cells[next] : memory[next];
        auto defense = cell.army;
        bool own = view->cells[next].owner == view->player;
        int arrival = plan.wait + static_cast<int>(plan.path.size());
        if (cell.owner >= 0) defense += growth(next, arrival - 1);
        bool first = plan.path.size() == 1;
        if (first && !safeDeparture(plan.path[0], next, plan.half, plan.wait)) return false;
        auto sent = first && plan.half ? plan.army / 2 : plan.army - 1;
        if (sent < 1 || (!own && sent <= defense)) return false;
        int from = plan.path.back();
        plan.placement += position(from, plan.army - sent) - position(from, plan.army);
        if (own) plan.placement -= position(next, defense);
        plan.army = sent + (own ? defense : -defense);
        plan.placement += position(next, plan.army);
        // growth after this move belongs to the captured cell and travels with the next move's army.
        plan.army += growth(next, arrival) - growth(next, arrival - 1);
        if (!own) {
            // compare income over the next 50 half-turns. enemy land also denies its next land-growth income.
            bool enemy = cell.owner == 1 - view->player;
            auto untilGrowth = 50 - view->tick % 50;
            plan.gain += 1 + (enemy ? 1 + std::abs(advantage) : 0);
            // a capture on the growth half-turn still changes which player receives that cell's new troop.
            if (static_cast<std::uint64_t>(arrival) <= untilGrowth) plan.gain += (enemy ? 2.0 : 1.0) / untilGrowth;
            if (producer(next)) plan.gain += 25;
            if (!seen[next]) plan.gain += 0.25 * (1 + alert) + std::abs(advantage) * trail[next];
            if (contact >= 0 && origins[next]) plan.gain += (1 + std::abs(advantage)) / (1 + std::abs(next / cols - contact / cols) + std::abs(next % cols - contact % cols));
        }
        plan.path.push_back(next);
        // placement measures changes to existing troops, rather than rewarding their collection a second time.
        plan.score = (plan.gain + plan.placement) / arrival;
        return true;
    }
    Plan toward(int target, bool winning) const {
        auto distance = distances(target);
        Plan best;
        for (int source : armies()) for (bool half : {false, true}) {
            if (distance[source] <= 0) continue;
            Plan plan{{source}, 0, view->cells[source].army, 0};
            plan.half = half;
            int pos = source;
            while (pos != target) {
                int choice = -1;
                std::int64_t largest = -1;
                for (int next : neighbors(pos)) {
                    if (next < 0 || distance[next] != distance[pos] - 1) continue;
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
    Action defend(int enemy) const {
        int deadline = home[enemy];
        auto target = toward(enemy, true);
        if (target.path.size() >= 2 && target.path.size() - 1 <= static_cast<std::size_t>(deadline)) {
            return move(target.path[0], target.path[1], target.half);
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
        std::vector<Plan> beam;
        // waiting competes with moving under the same score. only a producer can improve its expedition by waiting.
        for (int wait = 0; wait <= (producer(source) ? 6 : 0); ++wait) for (bool half : {false, true}) {
            beam.push_back({{source}, 0, view->cells[source].army + growth(source, wait), 0, half, wait});
        }
        Plan best;
        // bounded beam search rewards newly taken cells, while friendly transit only consumes time.
        for (int depth = 0; depth < 12 && !beam.empty(); ++depth) {
            std::vector<Plan> nextBeam;
            for (const auto& plan : beam) for (int next : neighbors(plan.path.back())) {
                if (!open(next) || std::find(plan.path.begin(), plan.path.end(), next) != plan.path.end()) continue;
                auto candidate = plan;
                if (!extend(candidate, next)) continue;
                if (candidate.score > best.score || (candidate.score == best.score && candidate.gain > best.gain)) best = candidate;
                nextBeam.push_back(std::move(candidate));
            }
            std::stable_sort(nextBeam.begin(), nextBeam.end(), [](const Plan& left, const Plan& right) {
                return left.score > right.score;
            });
            if (nextBeam.size() > 24) nextBeam.resize(24);
            beam = std::move(nextBeam);
        }
        return best;
    }
};
