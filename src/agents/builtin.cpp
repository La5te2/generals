#include "builtin.hpp"
#include <vector>

namespace Agents {
    Action builtin(const Observation& view, std::mt19937& random) {
        if (view.result != Phases::Ongoing || view.player < 0 || view.player > 1) return {};
        if (view.rows < 1 || view.cols < 1 || view.rows > dim || view.cols > dim) return {};
        std::vector<Action> actions(1); // Pass is available even when every army has size one.
        constexpr int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};
        for (int row = 0; row < view.rows; ++row) {
            for (int col = 0; col < view.cols; ++col) {
                const ViewCell& source = view.cells[row * view.cols + col];
                if (source.owner != view.player || source.army <= 1) continue;
                if (source.terrain != ViewTerrain::Plain && source.terrain != ViewTerrain::City &&
                    source.terrain != ViewTerrain::General) continue;
                for (int direction = 0; direction < 4; ++direction) {
                    int r = row + dr[direction], c = col + dc[direction];
                    if (r < 0 || c < 0 || r >= view.rows || c >= view.cols) continue;
                    ViewTerrain terrain = view.cells[r * view.cols + c].terrain;
                    // owned cells reveal every adjacent destination. city defenses affect combat, not legality.
                    if (terrain != ViewTerrain::Plain && terrain != ViewTerrain::City &&
                        terrain != ViewTerrain::General) continue;
                    actions.push_back({ActionType::Move, row, col, static_cast<Direction>(direction), false});
                    actions.push_back({ActionType::Move, row, col, static_cast<Direction>(direction), true});
                }
            }
        }
        std::uniform_int_distribution<std::size_t> choose(0, actions.size() - 1);
        return actions[choose(random)];
    }
}
