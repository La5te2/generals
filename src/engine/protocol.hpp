#pragma once

#include "actions.hpp"
#include "observe.hpp"

#include <istream>
#include <locale>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>

namespace Protocol {
    // sends player_id, height and width once when the strategy joins a new game.
    // returns false for invalid metadata or an output failure.
    inline bool writeInit(std::ostream& output, const Observation& view) {
        if (view.player != 0 && view.player != 1) return false;
        if (view.rows < 1 || view.rows > dim || view.cols < 1 || view.cols > dim) return false;

        // classic formatting keeps protocol integers decimal and free of digit grouping.
        std::ostringstream message;
        message.imbue(std::locale::classic());
        message << view.player << ' ' << view.rows << ' ' << view.cols << '\n';
        output << message.str() << std::flush;
        return static_cast<bool>(output);
    }

    // sends tick, own land and army, enemy land and army, then type, owner and army grids.
    // tick counts half-turns. owner codes are 0 for neutral or hidden, 1 for self and 2 for enemy.
    // returns false for a terminal observation, invalid data or an output failure.
    // after the game ends, the caller closes the strategy's input pipe to signal EOF.
    inline bool writeObservation(std::ostream& output, const Observation& view) {
        if (view.result != Phases::Ongoing) return false;
        if (view.player != 0 && view.player != 1) return false;
        if (view.rows < 1 || view.rows > dim || view.cols < 1 || view.cols > dim) return false;

        int player = view.player, opponent = 1 - player;
        std::ostringstream message;
        message.imbue(std::locale::classic());
        message << view.tick << ' ' << view.land[player] << ' ' << view.armies[player] << ' '
                << view.land[opponent] << ' ' << view.armies[opponent] << '\n';

        // assemble a complete observation before writing it to the strategy's stream.
        for (int grid = 0; grid < 3; ++grid) {
            for (int row = 0; row < view.rows; ++row) {
                for (int col = 0; col < view.cols; ++col) {
                    const ViewCell& cell = view.cells[row * view.cols + col];
                    bool hidden = cell.terrain == ViewTerrain::Fog || cell.terrain == ViewTerrain::Obstacle;
                    std::int64_t value = 0;
                    if (grid == 0) {
                        switch (cell.terrain) {
                            case ViewTerrain::Fog: value = 0; break;
                            case ViewTerrain::Plain: value = 1; break;
                            case ViewTerrain::Mountain: value = 2; break;
                            case ViewTerrain::City: value = 3; break;
                            case ViewTerrain::General: value = 4; break;
                            case ViewTerrain::Obstacle: value = 5; break;
                            default: return false;
                        }
                    } else if (grid == 1) {
                        if (!hidden) {
                            if (cell.owner == player) value = 1;
                            else if (cell.owner == opponent) value = 2;
                            else if (cell.owner != -1) return false;
                        }
                    } else if (!hidden) {
                        if (cell.army < 0) return false;
                        value = cell.army;
                    }
                    if (col > 0) message << ' ';
                    message << value;
                }
                message << '\n';
            }
        }
        output << message.str() << std::flush;
        return static_cast<bool>(output);
    }

    // reads exactly one line containing kind, row, col, direction and split.
    // move uses kind 0 and pass uses kind 1. pass ignores the other four integer values.
    // returns std::nullopt for a malformed or unsupported action, EOF or an input failure.
    // malformed lines are consumed, so the next call starts with the next reply.
    // the caller supplies a Pass for a rejected reply. rules.hpp checks move legality on the board.
    inline std::optional<Action> readAction(std::istream& input) {
        std::string line;
        if (!std::getline(input, line)) return std::nullopt;
        std::istringstream message(line);
        message.imbue(std::locale::classic());
        int kind, row, col, direction, split;
        if (!(message >> kind >> row >> col >> direction >> split)) return std::nullopt;
        message >> std::ws;
        if (!message.eof()) return std::nullopt;

        Action action;
        switch (kind) {
            case 0: action.type = ActionType::Move; break;
            case 1: action.type = ActionType::Pass; return action;
            default: return std::nullopt;
        }
        action.row = row;
        action.col = col;
        switch (direction) {
            case 0: action.direction = Direction::Up; break;
            case 1: action.direction = Direction::Down; break;
            case 2: action.direction = Direction::Left; break;
            case 3: action.direction = Direction::Right; break;
            default: return std::nullopt;
        }
        switch (split) {
            case 0: action.half = false; break;
            case 1: action.half = true; break;
            default: return std::nullopt;
        }
        return action;
    }
}
