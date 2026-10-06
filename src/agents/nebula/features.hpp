// observation memory gives training and deployed agents the same spatial and temporal inputs.
#pragma once

#include "engine/observe.hpp"
#include "engine/actions.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace Learning {
    constexpr int Channels = 38, History = 7, Window = 512, Actions = 9;

    inline Action decode(int index, int side) {
        int area = side * side, kind = index / area, pos = index % area;
        if (index < 0 || kind >= Actions) throw std::invalid_argument("Invalid action index");
        if (kind == 8) return {};
        return {ActionType::Move, pos / side, pos % side, static_cast<Direction>(kind % 4), kind >= 4};
    }

    class Features {
    public:
        explicit Features(int size) : side(size), area(checkedArea(size)), spatial(Channels * area),
            legal(Actions * area), temporal(2 * Window), lastOwn(area), lastEnemy(area), age(area) {}

        void reset() {
            for (auto buffer : {&spatial, &temporal, &lastOwn, &lastEnemy, &age}) std::fill(buffer->begin(), buffer->end(), 0.0f);
            std::fill(legal.begin(), legal.end(), std::uint8_t{0});
            tick = 0;
            initialized = false;
        }

        void update(const Observation& view) {
            if (view.rows > side || view.cols > side || view.rows <= 0 || view.cols <= 0) {
                throw std::invalid_argument("Board exceeds the model input size");
            }
            if (initialized && view.tick <= tick) throw std::invalid_argument("Observation ticks must increase");
            float elapsed = initialized ? static_cast<float>(view.tick - tick) : 1.0f;
            tick = view.tick;
            initialized = true;
            std::fill(legal.begin(), legal.end(), std::uint8_t{0});
            std::fill(legal.begin() + 8 * area, legal.end(), std::uint8_t{1});
            // move history back before writing the latest change. hidden armies contribute zero to current channels.
            for (int history = History - 1; history > 0; --history) {
                for (int pos = 0; pos < area; ++pos) {
                    at(24 + history, pos) = at(23 + history, pos);
                    at(31 + history, pos) = at(30 + history, pos);
                }
            }
            for (int pos = 0; pos < area; ++pos) {
                int row = pos / side, col = pos % side;
                bool inside = row < view.rows && col < view.cols;
                ViewCell cell = inside ? view.cells[row * view.cols + col] : ViewCell{ViewTerrain::Obstacle, -1, 0};
                bool own = cell.owner == view.player, enemy = cell.owner == 1 - view.player;
                bool visible = cell.terrain != ViewTerrain::Fog && cell.terrain != ViewTerrain::Obstacle;
                float army = static_cast<float>(cell.army);
                float ours = own ? army : 0.0f, theirs = enemy ? army : 0.0f;
                at(0, pos) = army; at(1, pos) = ours; at(2, pos) = theirs;
                at(3, pos) = cell.owner < 0 && visible ? army : 0.0f;
                at(6, pos) = std::max(at(6, pos), static_cast<float>(cell.terrain == ViewTerrain::General));
                at(7, pos) = std::max(at(7, pos), static_cast<float>(cell.terrain == ViewTerrain::City));
                at(8, pos) = std::max(at(8, pos), static_cast<float>(cell.terrain == ViewTerrain::Mountain));
                at(9, pos) = static_cast<float>(cell.owner < 0 && visible && cell.terrain != ViewTerrain::Mountain);
                at(10, pos) = static_cast<float>(own); at(11, pos) = static_cast<float>(enemy);
                at(12, pos) = static_cast<float>(cell.terrain == ViewTerrain::Fog);
                at(13, pos) = static_cast<float>(cell.terrain == ViewTerrain::Obstacle);
                at(14, pos) = static_cast<float>(view.tick); at(15, pos) = static_cast<float>(view.tick % 50) / 50.0f;
                at(16, pos) = static_cast<float>(view.land[view.player]);
                at(17, pos) = static_cast<float>(view.armies[view.player]);
                at(18, pos) = static_cast<float>(view.land[1 - view.player]);
                at(19, pos) = static_cast<float>(view.armies[1 - view.player]);
                if (theirs > 0) { at(20, pos) = theirs; age[pos] = 0; }
                else age[pos] += elapsed;
                at(21, pos) = std::log1p(age[pos]) / 5.0f;
                at(22, pos) = static_cast<float>(col) / (side - 1);
                at(23, pos) = static_cast<float>(row) / (side - 1);
                at(24, pos) = ours - lastOwn[pos]; at(31, pos) = theirs - lastEnemy[pos];
                lastOwn[pos] = ours; lastEnemy[pos] = theirs;

                if (own || enemy) {
                    // enemy sight is inferred from visible enemy territory, rather than the hidden board.
                    for (int r = std::max(0, row - 1); r <= std::min(side - 1, row + 1); ++r) {
                        for (int c = std::max(0, col - 1); c <= std::min(side - 1, col + 1); ++c) {
                            at(own ? 4 : 5, r * side + c) = 1;
                        }
                    }
                }
                if (!own || cell.army < 2) continue;
                constexpr int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};
                for (int direction = 0; direction < 4; ++direction) {
                    int r = row + dr[direction], c = col + dc[direction];
                    if (r < 0 || c < 0 || r >= view.rows || c >= view.cols) continue;
                    if (view.cells[r * view.cols + c].terrain == ViewTerrain::Mountain) continue;
                    legal[direction * area + pos] = 1;
                    legal[(direction + 4) * area + pos] = 1;
                }
            }
            // revealed padding is remembered as mountain. distant padding has the same symbol as a hidden structure.
            for (int row = 0; row < side; ++row) for (int col = 0; col < side; ++col) {
                if (row < view.rows && col < view.cols) continue;
                int pos = row * side + col;
                at(8, pos) = at(4, pos);
                at(13, pos) = 1 - at(8, pos);
            }
            for (int channel = 0; channel < 2; ++channel) {
                auto begin = temporal.begin() + channel * Window;
                std::move(begin + 1, begin + Window, begin);
                *(begin + Window - 1) = static_cast<float>(channel == 0 ? view.armies[1 - view.player] : view.land[1 - view.player]);
            }
        }

        int side, area;
        std::vector<float> spatial;
        std::vector<std::uint8_t> legal;
        std::vector<float> temporal;

    private:
        static int checkedArea(int size) {
            if (size < 2 || size > dim) throw std::invalid_argument("Invalid observation size");
            return size * size;
        }
        std::vector<float> lastOwn, lastEnemy, age;
        std::uint64_t tick = 0;
        bool initialized = false;
        float& at(int channel, int pos) { return spatial[channel * area + pos]; }
    };
}
