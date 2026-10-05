#pragma once

#include <cstdint>

enum class ActionType : std::uint8_t { 
    Move = 0, Pass = 1
};
enum class Direction : std::uint8_t { 
    Up = 0, Down = 1, Left = 2, Right = 3
};

struct Action {
    ActionType type = ActionType::Pass;
    int row = 0, col = 0;
    Direction direction = Direction::Up;
    bool half = false;
};
