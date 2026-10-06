// human input: keep selection and movement controls separate from window navigation and game rules.
#include "controller.hpp"
#include "local.hpp"
#include "scene.hpp"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace NEBULA {
    void Controller::reset(int player, const Observation& view) {
        side = player;
        selected = -1;
        halfArmy = false;
        if (side >= 0) general(view);
    }

    void Controller::general(const Observation& view) {
        for (int cell = 0; cell < view.rows * view.cols; ++cell) {
            if (view.cells[cell].owner == side && view.cells[cell].terrain == ViewTerrain::General) {
                selected = cell;
                halfArmy = false;
                return;
            }
        }
    }

    void Controller::sync(const LocalSnapshot& snapshot) {
        if (side < 0) return;
        const auto& view = (*snapshot.views)[side];
        // a planned endpoint can still be neutral. after the queue finishes, selection follows actual ownership.
        if (snapshot.state != LocalState::Active || (selected >= 0 && snapshot.queued[side].empty() &&
            view.cells[selected].owner != side)) {
            selected = -1;
            halfArmy = false;
        }
    }

    void Controller::move(LocalMatch& match, const LocalSnapshot& snapshot, Direction direction) {
        if (side < 0 || selected < 0 || snapshot.state != LocalState::Active) return;
        const auto& view = (*snapshot.views)[side];
        int row = selected / view.cols, col = selected % view.cols;
        int nextRow = row, nextCol = col;
        switch (direction) {
            case Direction::Up: --nextRow; break;
            case Direction::Down: ++nextRow; break;
            case Direction::Left: --nextCol; break;
            case Direction::Right: ++nextCol; break;
        }
        if (nextRow < 0 || nextRow >= view.rows || nextCol < 0 || nextCol >= view.cols) return;
        int next = nextRow * view.cols + nextCol;
        // fogged obstacles can be cities. only a visible mountain proves that movement is blocked.
        if (view.cells[next].terrain == ViewTerrain::Mountain) return;
        if (view.cells[selected].owner != side && snapshot.queued[side].empty()) return;
        if (match.enqueue(side, {ActionType::Move, row, col, direction, halfArmy})) {
            selected = next;
            halfArmy = false;
        }
    }

    void Controller::click(LocalMatch& match, const LocalSnapshot& snapshot, double x, double y, int width, int height) {
        if (side < 0 || snapshot.state != LocalState::Active) return;
        const auto& view = (*snapshot.views)[side];
        Rect board = boardArea(view.rows, view.cols, width, height);
        if (!board.contains(x, y)) return;
        float size = board.width / view.cols;
        int row = static_cast<int>((y - board.y) / size), col = static_cast<int>((x - board.x) / size);
        if (row >= view.rows || col >= view.cols) return;
        int cell = row * view.cols + col;
        if (cell == selected) { halfArmy = !halfArmy; return; }
        if (selected >= 0) {
            int dr = row - selected / view.cols, dc = col - selected % view.cols;
            if (dr == -1 && dc == 0) { move(match, snapshot, Direction::Up); return; }
            if (dr == 1 && dc == 0) { move(match, snapshot, Direction::Down); return; }
            if (dr == 0 && dc == -1) { move(match, snapshot, Direction::Left); return; }
            if (dr == 0 && dc == 1) { move(match, snapshot, Direction::Right); return; }
        }
        if (view.cells[cell].owner == side) { selected = cell; halfArmy = false; }
    }

    void Controller::cancel(LocalMatch& match, bool all) {
        if (auto removed = match.cancel(side, all)) {
            const auto snapshot = match.snapshot();
            selected = removed->row * (*snapshot.views)[side].cols + removed->col;
        }
        halfArmy = false;
    }

    void Controller::key(LocalMatch& match, const LocalSnapshot& snapshot, int key, int action, int mods) {
        if (side < 0 || snapshot.state != LocalState::Active ||
            (action != GLFW_PRESS && action != GLFW_REPEAT) || (mods & (GLFW_MOD_CONTROL | GLFW_MOD_SUPER | GLFW_MOD_ALT))) return;
        switch (key) {
            case GLFW_KEY_W: case GLFW_KEY_UP: move(match, snapshot, Direction::Up); break;
            case GLFW_KEY_S: case GLFW_KEY_DOWN: move(match, snapshot, Direction::Down); break;
            case GLFW_KEY_A: case GLFW_KEY_LEFT: move(match, snapshot, Direction::Left); break;
            case GLFW_KEY_D: case GLFW_KEY_RIGHT: move(match, snapshot, Direction::Right); break;
            case GLFW_KEY_E: cancel(match, false); break;
            case GLFW_KEY_Q: cancel(match, true); break;
            case GLFW_KEY_G: general((*snapshot.views)[side]); break;
            case GLFW_KEY_LEFT_SHIFT: case GLFW_KEY_RIGHT_SHIFT:
                if (action == GLFW_PRESS && selected >= 0) halfArmy = !halfArmy;
                break;
        }
    }
}
