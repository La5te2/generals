#pragma once

#include "states.hpp"

enum class ViewTerrain : std::uint8_t {
    Fog = 0, Plain = 1, Mountain = 2, City = 3, General = 4, Obstacle = 5
    // hidden mountains and cities both appear as Obstacle.
};

struct ViewCell {
    ViewTerrain terrain = ViewTerrain::Fog;
    std::int8_t owner = -1;
    std::int64_t army = 0;
    // owner and army describe the cell only while it is visible.
};

struct Observation {
    int rows = 0, cols = 0;
    int player = 0; // 0 for red, 1 for blue.
    std::uint64_t tick = 0;
    Phases result = Phases::Ongoing;

    std::array<ViewCell, Capacity> cells{};
    std::array<std::int64_t, 2> armies{};
    std::array<int, 2> land{};
    // armies and land contain the public totals for red and blue.
};

inline Observation observe(const States& state, int player) {
    Observation view;
    view.rows = state.board.rows();
    view.cols = state.board.cols();
    view.player = player;
    view.tick = state.tick;
    view.result = state.result;

    // each owned cell reveals itself and its eight neighbors, even with zero army.
    std::array<bool, Capacity> visible{};
    for (int row = 0; row < view.rows; ++row) {
        for (int col = 0; col < view.cols; ++col) {
            if (state.board.at(row, col).owner != player) continue;
            for (int r = row - 1; r <= row + 1; ++r) {
                for (int c = col - 1; c <= col + 1; ++c) {
                    if (state.board.contains(r, c)) visible[r * view.cols + c] = true;
                }
            }
        }
    }

    // visible cells show their current terrain, owner and army.
    // hidden mountains and cities appear as Obstacle. all other hidden cells appear as Fog.
    // hidden cells keep the default owner and army values.
    for (int row = 0; row < state.board.rows(); ++row) {
        for (int col = 0; col < state.board.cols(); ++col) {
            const Cell& cell = state.board.at(row, col);
            ViewCell& viewCell = view.cells[row * state.board.cols() + col];
            if (visible[row * view.cols + col]) {
                switch (cell.terrain) {
                    case Terrain::Plain: viewCell.terrain = ViewTerrain::Plain; break;
                    case Terrain::Mountain: viewCell.terrain = ViewTerrain::Mountain; break;
                    case Terrain::City: viewCell.terrain = ViewTerrain::City; break;
                    case Terrain::General: viewCell.terrain = ViewTerrain::General; break;
                }
                viewCell.owner = cell.owner;
                viewCell.army = cell.army;
            } else if (cell.terrain == Terrain::Mountain || cell.terrain == Terrain::City) {
                viewCell.terrain = ViewTerrain::Obstacle;
            }
        }
    }

    // calculate public totals for armies and land.
    for (int row = 0; row < state.board.rows(); ++row) {
        for (int col = 0; col < state.board.cols(); ++col) {
            const Cell& cell = state.board.at(row, col);
            if (cell.owner == 0 || cell.owner == 1) {
                view.armies[cell.owner] += cell.army;
                ++view.land[cell.owner];
            }
        }
    }

    return view;
}
