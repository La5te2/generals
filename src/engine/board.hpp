#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

inline constexpr int dim = 40; // Max Side Length
inline constexpr int Capacity = dim * dim;

enum class Terrain : std::uint8_t {
    Plain = 1, Mountain = 2, City = 3, General = 4
};

struct Cell {
    std::int64_t army = 0;
    std::int8_t owner = -1; 
    Terrain terrain = Terrain::Plain;
    // owner: -1 = neutral, 0 = red, 1 = blue.
};

class Board {
public:
    Board(int rows, int cols) : R(rows), C(cols) {
        if (rows < 1 || rows > dim || cols < 1 || cols > dim) {
            throw std::out_of_range("Board dimensions out of range");
        }
    }

    int rows() const noexcept { return R; }
    int cols() const noexcept { return C; }
    int size() const noexcept { return R * C; }
    bool contains(int row, int col) const noexcept {
        return row >= 0 && row < R && col >= 0 && col < C;
    }

    // at: Returns a reference to the cell at the specified coordinates.
    // The caller is responsible for ensuring that the coordinates are valid.
    // Throws std::out_of_range if the coordinates are invalid.
    Cell& at(int row, int col) { return cells[index(row, col)]; }
    const Cell& at(int row, int col) const { return cells[index(row, col)]; }

private:
    int index(int row, int col) const {
        if (!contains(row, col)) {
            throw std::out_of_range("Cell coordinates are outside the board");
        }
        return row * C + col;
    }

    int R, C;
    // Active cells occupy [0, R * C), using C as the row stride.
    std::array<Cell, Capacity> cells{};
};
