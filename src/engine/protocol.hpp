// shared wire format: exchange observations and actions independently of files and processes.
#pragma once

#include "actions.hpp"
#include "observe.hpp"

#include <istream>
#include <charconv>
#include <locale>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>

namespace Protocol {
    // one process plays one game. initialization precedes observations, and EOF ends the game.
    // each observation receives one action in response. stdout carries protocol data, stderr diagnostics.
    // a slow strategy receives the latest pending observation after replying. its next tick can therefore skip ahead.
    struct Init {
        int player = 0, rows = 0, cols = 0;
    };

    template<class Integer>
    inline bool integer(std::istream& input, Integer& value) {
        std::string token;
        if (!(input >> token)) return false;
        auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
        return error == std::errc{} && end == token.data() + token.size();
    }

    inline bool valid(const Init& init) {
        return (init.player == 0 || init.player == 1) && init.rows >= 1 && init.rows <= dim &&
               init.cols >= 1 && init.cols <= dim;
    }

    // initialization: player rows cols. player is 0 for red and 1 for blue.
    inline bool writeInit(std::ostream& output, const Init& init) {
        if (!valid(init)) return false;

        // classic formatting keeps protocol integers decimal and free of digit grouping.
        std::ostringstream message;
        message.imbue(std::locale::classic());
        message << init.player << ' ' << init.rows << ' ' << init.cols << '\n';
        output << message.str() << std::flush;
        return static_cast<bool>(output);
    }

    inline std::optional<Init> readInit(std::istream& input) {
        std::string line;
        if (!std::getline(input, line)) return std::nullopt;
        std::istringstream message(line);
        message.imbue(std::locale::classic());
        Init init;
        if (!integer(message, init.player) || !integer(message, init.rows) || !integer(message, init.cols) || !valid(init)) {
            return std::nullopt;
        }
        message >> std::ws;
        if (!message.eof()) return std::nullopt;
        return init;
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

    // decode relative owners back to red/blue indices. each matrix row contains exactly cols integers.
    inline std::optional<Observation> readObservation(std::istream& input, const Init& init) {
        if (!valid(init)) return std::nullopt;
        std::string line;
        if (!std::getline(input, line)) return std::nullopt;
        std::istringstream header(line);
        header.imbue(std::locale::classic());
        Observation view;
        view.player = init.player;
        view.rows = init.rows;
        view.cols = init.cols;
        int player = init.player, opponent = 1 - player;
        if (!integer(header, view.tick) || !integer(header, view.land[player]) || !integer(header, view.armies[player]) ||
            !integer(header, view.land[opponent]) || !integer(header, view.armies[opponent])) return std::nullopt;
        header >> std::ws;
        if (!header.eof()) return std::nullopt;
        for (int index = 0; index < 2; ++index) {
            if (view.land[index] < 0 || view.land[index] > init.rows * init.cols || view.armies[index] < 0) return std::nullopt;
        }
        for (int grid = 0; grid < 3; ++grid) {
            for (int row = 0; row < init.rows; ++row) {
                if (!std::getline(input, line)) return std::nullopt;
                std::istringstream values(line);
                values.imbue(std::locale::classic());
                for (int col = 0; col < init.cols; ++col) {
                    std::int64_t value = 0;
                    if (!integer(values, value)) return std::nullopt;
                    ViewCell& cell = view.cells[row * init.cols + col];
                    if (grid == 0) {
                        switch (value) {
                            case 0: cell.terrain = ViewTerrain::Fog; break;
                            case 1: cell.terrain = ViewTerrain::Plain; break;
                            case 2: cell.terrain = ViewTerrain::Mountain; break;
                            case 3: cell.terrain = ViewTerrain::City; break;
                            case 4: cell.terrain = ViewTerrain::General; break;
                            case 5: cell.terrain = ViewTerrain::Obstacle; break;
                            default: return std::nullopt;
                        }
                    } else {
                        bool hidden = cell.terrain == ViewTerrain::Fog || cell.terrain == ViewTerrain::Obstacle;
                        if (hidden && value != 0) return std::nullopt;
                        if (grid == 1) {
                            switch (value) {
                                case 0: cell.owner = -1; break;
                                case 1: cell.owner = static_cast<std::int8_t>(player); break;
                                case 2: cell.owner = static_cast<std::int8_t>(opponent); break;
                                default: return std::nullopt;
                            }
                        } else {
                            if (value < 0) return std::nullopt;
                            cell.army = value;
                        }
                    }
                }
                values >> std::ws;
                if (!values.eof()) return std::nullopt;
            }
        }
        return view;
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

    // writes kind, row, col, direction and split. the caller flushes after sending an action through a pipe.
    inline bool writeAction(std::ostream& output, const Action& action) {
        std::ostringstream message;
        message.imbue(std::locale::classic());
        if (action.type == ActionType::Pass) message << "1 0 0 0 0\n";
        else if (action.type == ActionType::Move) {
            int direction = 0;
            switch (action.direction) {
                case Direction::Up: direction = 0; break;
                case Direction::Down: direction = 1; break;
                case Direction::Left: direction = 2; break;
                case Direction::Right: direction = 3; break;
                default: return false;
            }
            message << "0 " << action.row << ' ' << action.col << ' ' << direction << ' ' << int(action.half) << '\n';
        } else return false;
        output << message.str();
        return static_cast<bool>(output);
    }

}
