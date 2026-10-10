#pragma once

#include "actions.hpp"
#include "states.hpp"

#include <array>

inline constexpr std::uint64_t tickLimit = 50000;
inline constexpr std::uint32_t idleLimit = 2000;

// checks if the action is legitimate for the given player and state
inline bool legit(const States& state, int player, const Action& action) { 
    bool legitimacy = true;
    legitimacy &= (player == 0 || player == 1);
    legitimacy &= (state.result == Phases::Ongoing);
    if (action.type == ActionType::Pass) return legitimacy; // passing requires a valid player and an ongoing game.
    if (action.type != ActionType::Move) return false; // only Pass and Move are valid action types.

    // validate coordinates before indexing or calculating an adjacent cell.
    if (!state.board.contains(action.row, action.col)) {
        return false;
    }
    // validate direction before calculating an adjacent cell.
    if (action.direction != Direction::Up && action.direction != Direction::Down &&
        action.direction != Direction::Left && action.direction != Direction::Right) {
        return false;
    }

    const Cell& source = state.board.at(action.row, action.col);
    auto offset = directionOffsets[static_cast<int>(action.direction)];
    int tr = action.row + offset.row, tc = action.col + offset.col;

    legitimacy &= (source.owner == player);
    legitimacy &= (source.army > 1);
    legitimacy &= (source.terrain != Terrain::Mountain);
    legitimacy &= state.board.contains(tr, tc);

    // city and general defenses only affect combat outcomes, not action legality.
    return legitimacy && state.board.at(tr, tc).terrain != Terrain::Mountain;
}

// executes the action on the state for the given player, modifying the state in place, 
// and returns true if the action was executed successfully, false otherwise
// this function settles one action. step() handles move order, growth and game results.
inline bool execute(States& state, int player, const Action& action) {
    if (!legit(state, player, action))  return false; // action is not legitimate, do not modify the state.
    if (action.type == ActionType::Pass) return true; // a Pass leaves the board unchanged.
    auto offset = directionOffsets[static_cast<int>(action.direction)];
    int tr = action.row + offset.row, tc = action.col + offset.col;
    Cell& source = state.board.at(action.row, action.col);
    Cell& target = state.board.at(tr, tc);
    std::int64_t army = action.half ? source.army / 2 : source.army - 1;
    source.army -= army;

    if (target.owner == player) {
        target.army += army; // friendly movement combines the two armies.
    } else if (army > target.army) { // target.owner != player and army > target.army
        target.army = army - target.army;
        target.owner = static_cast<std::int8_t>(player);
    } else { // target.owner != player and army <= target.army
        target.army -= army; // a tie leaves the defender owning a zero-army cell.
    }
    return true;
}

// applies growth once after step advances tick, including the final winning step.
inline void grow(States& state) {
    if (state.tick == 0) return; // initial armies belong to board setup.
    bool recruit = state.tick % 2 == 0; // one full turn per two half-turn ticks.
    bool troops = state.tick % 50 == 0; // land grows every 25 full turns.
    if (!recruit && !troops) return;

    for (int row = 0; row < state.board.rows(); ++row) {
        for (int col = 0; col < state.board.cols(); ++col) {
            Cell& cell = state.board.at(row, col);
            if (cell.owner < 0 || cell.terrain == Terrain::Mountain) continue;
            if (recruit && (cell.terrain == Terrain::City || cell.terrain == Terrain::General)) {
                ++cell.army;
            }
            if (troops) ++cell.army;
        }
    }
}

// resolves a half-turn
inline void step(States& state, const std::array<Action, 2>& actions) {
    if (state.result != Phases::Ongoing) return;

    // Only a possible timeout needs pre-move totals; moves and growth must not affect this comparison.
    std::array<std::int64_t, 2> totals{};
    std::array<int, 2> land{};
    if (state.tick > tickLimit || state.idle >= idleLimit) {
        for (int row = 0; row < state.board.rows(); ++row) {
            for (int col = 0; col < state.board.cols(); ++col) {
                const Cell& cell = state.board.at(row, col);
                if (cell.owner == 0 || cell.owner == 1) {
                    totals[cell.owner] += cell.army;
                    ++land[cell.owner];
                }
            }
        }
    }

    // actions[0] belongs to red, actions[1] belongs to blue. 
    // each player may pass or move once per half-turn.
    // board storage stays fixed during a step.
    std::array<Cell*, 2> source{}, target{};
    for (int player = 0; player < 2; ++player) {
        const Action& action = actions[player];
        if (action.type != ActionType::Move || !legit(state, player, action)) continue;
        auto offset = directionOffsets[static_cast<int>(action.direction)];
        source[player] = &state.board.at(action.row, action.col);
        target[player] = &state.board.at(action.row + offset.row, action.col + offset.col);
    }

    // determines move order from the board before either player acts.
    // equal priorities: red first on even ticks, blue first on odd ticks.
    int first = static_cast<int>(state.tick % 2); 
    bool both = source[0] != nullptr && source[1] != nullptr;
    if (both) {
        bool redChases = target[0] == source[1] && target[1] != source[0];
        bool blueChases = target[1] == source[0] && target[0] != source[1];
        bool redDefends = target[0]->owner == 0;
        bool blueDefends = target[1]->owner == 1;
        bool redGeneral = target[0]->terrain == Terrain::General;
        bool blueGeneral = target[1]->terrain == Terrain::General;

        // determines which move executes first by checking these rules in order:
        // 1. for A -> B and B -> C with C != A, execute A -> B first.
        //    opposite moves A -> B and B -> A use the rules below.
        // 2. if exactly one move targets its player's own cell, execute that move first.
        // 3. if exactly one move targets a general, execute the other move first.
        // 4. if source armies differ, execute the move from the larger army first.
        //    compare the full source armies, including for half-army moves.
        // if all priorities are equal, keep the tick-based order above.
        if (redChases) first = 0;
        else if (blueChases) first = 1;
        else if (redDefends != blueDefends) first = redDefends ? 0 : 1;
        else if (redGeneral != blueGeneral) first = redGeneral ? 1 : 0;
        else if (source[0]->army != source[1]->army) first = source[0]->army > source[1]->army ? 0 : 1;
    }

    bool traded = false; // a special case where both generals are captured in a single step, leaving the game ongoing.
    if (both && target[0]->terrain == Terrain::General && target[0]->owner == 1 &&
        target[1]->terrain == Terrain::General && target[1]->owner == 0) {
        std::array<std::int64_t, 2> army{}, defense{};
        for (int player = 0; player < 2; ++player) {
            army[player] = actions[player].half ? source[player]->army / 2 : source[player]->army - 1;
            defense[player] = target[player]->army;
        }

        // the trading happens when both captures are strictly successful. 
        // a tie keeps the defender's general(unsuccessful capture).
        // when the trading happens, subtract each attacking army from its source cell first,
        // all occupied cells except the two general cells then switch owners,
        // and their remaining armies are halved, rounding up.
        if (army[0] > defense[0] && army[1] > defense[1]) {
            for (int player = 0; player < 2; ++player) source[player]->army -= army[player];
            for (int row = 0; row < state.board.rows(); ++row) {
                for (int col = 0; col < state.board.cols(); ++col) {
                    Cell& cell = state.board.at(row, col);
                    if (cell.owner < 0 || &cell == target[0] || &cell == target[1]) continue;
                    cell.owner = static_cast<std::int8_t>(1 - cell.owner);
                    cell.army = cell.army / 2 + cell.army % 2;
                }
            }
            for (int player = 0; player < 2; ++player) {
                target[player]->owner = static_cast<std::int8_t>(player);
                target[player]->army = army[player] - defense[player];
            }
            traded = true; // marks both actions as already resolved by the general trade.
        }
    }

    bool moved = traded;
    if (!traded) { // normal step resolution, including the case where only one player moves.
        for (int order = 0; order < 2; ++order) {
            int player = (first + order) % 2;
            if (source[player] == nullptr) continue;
            int defender = target[player]->owner;
            bool general = target[player]->terrain == Terrain::General && defender == 1 - player;

            // execute() rechecks whether the source cell still belongs to this player and has at least two troops.
            // the first player's move may have captured that cell or reduced its army before the second player acts.
            if (!execute(state, player, actions[player])) continue;
            moved = true;
            if (general && target[player]->owner == player) {
                target[player]->terrain = Terrain::City;
                for (int row = 0; row < state.board.rows(); ++row) {
                    for (int col = 0; col < state.board.cols(); ++col) {
                        Cell& cell = state.board.at(row, col);
                        if (cell.owner != defender) continue;
                        cell.owner = static_cast<std::int8_t>(player);
                        cell.army = cell.army / 2 + cell.army % 2;
                    }
                }
                // the captured general already belongs to the winner, preserving the attacking survivors.
                state.result = player == 0 ? Phases::RedWin : Phases::BlueWin;
                break;
            }
        }
    }

    bool expired = state.tick > tickLimit;
    if (moved && !expired) state.idle = 0;
    else ++state.idle;

    // after more than 2000 idle half-turns or a starting tick above 50000, compare the starting scores.
    // the larger army wins, then the larger land count. blue wins an exact tie.
    // this ending leaves land ownership and terrain unchanged, while a general capture keeps its winner.
    if (state.result == Phases::Ongoing && (expired || state.idle > idleLimit)) {
        int winner = 1;
        if (totals[0] > totals[1] || (totals[0] == totals[1] && land[0] > land[1])) winner = 0;
        state.result = winner == 0 ? Phases::RedWin : Phases::BlueWin;
    }
    ++state.tick; // the final winning step also completes its scheduled growth.
    grow(state);
}
