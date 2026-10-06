// human.hpp defines keyboard and mouse controls for one player in a local or online match.
// it stores the selected cell and full or half army setting so input handling and drawing share these choices.
#pragma once

#include "engine/actions.hpp"

struct Observation;

namespace NEBULA {
    class PlayerInput;
    struct MatchSnapshot;

    class Human {
    public:
        // assign red (0) or blue (1), select their general, and choose full-army moves.
        // player -1 disables player input and clears the selection.
        void reset(int player, const Observation& view);

        // clear the selection when the match ends or the selected cell leaves the board.
        // also clear it when the queue is empty and the cell is neutral or belongs to another player.
        void sync(const MatchSnapshot& snapshot);

        // a board click selects an owned cell, queues a move to an adjacent cell,
        // or toggles the next move's army size when clicking the selected cell again.
        void click(PlayerInput& match, const MatchSnapshot& snapshot, double x, double y, int width, int height);

        // use WASD and arrow keys to queue moves. E cancels the last queued move, Q cancels the whole queue,
        // G selects the general, and Shift toggles full or half army for the next move.
        void key(PlayerInput& match, const MatchSnapshot& snapshot, int key, int action, int mods);

        // clear the selection and restore full-army input. queued moves remain with the session.
        void deselect() { selected = -1; halfArmy = false; }

        // the controlled side: 0 for red, 1 for blue, or -1 when player input is disabled.
        int player() const { return side; }
        // row * cols + col for the selected cell or queued route endpoint. -1 means an empty selection.
        int selection() const { return selected; }
        // whether the next queued move sends half of the source cell's army.
        bool half() const { return halfArmy; }

    private:
        // select the controlled player's general if it appears in the observation, and restore full-army input.
        void general(const Observation& view);
        // submit one move from the selected cell. acceptance advances the selection to its destination.
        void move(PlayerInput& match, const MatchSnapshot& snapshot, Direction direction);
        // cancel the last queued move or, with all=true, the whole queue.
        // return the selection to the source of the earliest removed move, and restore full-army input.
        void cancel(PlayerInput& match, const MatchSnapshot& snapshot, bool all);

        int side = -1;
        int selected = -1; // row * cols + col, including the endpoint of a queued route.
        bool halfArmy = false; // send half on the next move, then return to full moves.
    };
}
