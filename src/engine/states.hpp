#pragma once

#include "board.hpp"
#include <cstdint>

enum class Phases : std::uint8_t {
    Ongoing, RedWin, BlueWin, Draw
};
// RedWin corresponds to player 0, BlueWin to player 1.

struct States {
    Board board;
    std::uint64_t tick = 0;
    std::uint32_t idle = 0; // consecutive half-turns without an executed move.
    Phases result = Phases::Ongoing;
    
    // Construct an empty board and fill starting positions during setup.
    States(int rows, int cols) : board(rows, cols) {}
};
