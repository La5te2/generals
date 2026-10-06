// human input: turn board selections and directional input into queued player actions.
#pragma once

#include "engine/actions.hpp"

struct Observation;

namespace NEBULA {
    class LocalMatch;
    struct LocalSnapshot;

    class Controller {
    public:
        // player -1 selects spectator controls. a human starts with their general selected.
        void reset(int player, const Observation& view);
        void sync(const LocalSnapshot& snapshot);
        void click(LocalMatch& match, const LocalSnapshot& snapshot, double x, double y, int width, int height);
        void key(LocalMatch& match, const LocalSnapshot& snapshot, int key, int action, int mods);
        void deselect() { selected = -1; halfArmy = false; }

        int player() const { return side; }
        int selection() const { return selected; }
        bool half() const { return halfArmy; }

    private:
        void general(const Observation& view);
        void move(LocalMatch& match, const LocalSnapshot& snapshot, Direction direction);
        void cancel(LocalMatch& match, bool all);

        int side = -1;
        int selected = -1; // row * cols + col, including the endpoint of a queued route.
        bool halfArmy = false; // send half on the next move, then return to full moves.
    };
}
