// run exported GenFormer weights on CPU as an independent agent using the shared observation/action protocol.
// the model owns observation history and action masking; this executable handles process input and output only.
#include "engine/protocol.hpp"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4267 4702)
#endif
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/csrc/jit/serialization/import.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: simple MODEL.pt\n"; return 1; }
    try {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        at::NoGradGuard inference;
        auto model = torch::jit::load(argv[1], at::kCPU);
        model.eval();
        int side = static_cast<int>(model.attr("side").toInt());
        if (side < 1 || side > dim) throw std::invalid_argument("Invalid model input size");
        auto init = Protocol::readInit(std::cin);
        if (!init) throw std::invalid_argument("Invalid initialization");
        if (init->rows > side || init->cols > side) throw std::invalid_argument("Board exceeds the model input size");
        model.get_method("reset")({init->rows, init->cols});
        int area = side * side;
        auto board = at::zeros({1, 3, side, side}, at::kFloat);
        auto stats = at::zeros({1, 5}, at::kFloat);
        while (std::cin.peek() != std::char_traits<char>::eof()) {
            auto view = Protocol::readObservation(std::cin, *init);
            if (!view) throw std::invalid_argument("Invalid observation");
            float* data = board.data_ptr<float>();
            for (int row = 0; row < view->rows; ++row) for (int col = 0; col < view->cols; ++col) {
                const auto& cell = view->cells[row * view->cols + col];
                int pos = row * side + col;
                switch (cell.terrain) {
                    case ViewTerrain::Fog: data[pos] = 0; break;
                    case ViewTerrain::Plain: data[pos] = 1; break;
                    case ViewTerrain::Mountain: data[pos] = 2; break;
                    case ViewTerrain::City: data[pos] = 3; break;
                    case ViewTerrain::General: data[pos] = 4; break;
                    case ViewTerrain::Obstacle: data[pos] = 5; break;
                    default: throw std::invalid_argument("Invalid terrain");
                }
                data[area + pos] = cell.owner < 0 ? 0.0f : cell.owner == init->player ? 1.0f : 2.0f;
                data[2 * area + pos] = static_cast<float>(cell.army);
            }
            float* values = stats.data_ptr<float>();
            values[0] = static_cast<float>(view->tick);
            values[1] = static_cast<float>(view->land[init->player]);
            values[2] = static_cast<float>(view->armies[init->player]);
            values[3] = static_cast<float>(view->land[1 - init->player]);
            values[4] = static_cast<float>(view->armies[1 - init->player]);
            auto result = model.forward({board, stats}).toTuple();
            auto logits = result->elements()[0].toTensor();
            if (logits.dim() != 2 || logits.size(0) != 1 || logits.size(1) != 8 * area + 1 ||
                !at::isfinite(logits).all().item<bool>()) throw std::runtime_error("Invalid model action scores");
            int index = logits.argmax(1).item<int>();
            Action action;
            if (index < 8 * area) {
                int kind = index / area, pos = index % area;
                action = {ActionType::Move, pos / side, pos % side, static_cast<Direction>(kind % 4), kind >= 4};
            }
            model.get_method("submitted")({index});
            if (!Protocol::writeAction(std::cout, action) || !(std::cout << std::flush)) return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "simple: " << error.what() << '\n';
        return 1;
    }
}
