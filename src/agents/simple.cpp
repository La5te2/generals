// standalone strategy: exchange observations and planned actions through the standard streams.
#include "simple.hpp"
#include "engine/protocol.hpp"
#include <iostream>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    auto init = Protocol::readInit(std::cin);
    if (!init) { std::cerr << "Invalid initialization\n"; return 1; }
    Simple strategy;
    while (std::cin.peek() != std::char_traits<char>::eof()) {
        auto view = Protocol::readObservation(std::cin, *init);
        if (!view) { std::cerr << "Invalid observation\n"; return 1; }
        if (!Protocol::writeAction(std::cout, strategy.act(*view))) return 1;
        if (!(std::cout << std::flush)) return 1;
    }
    return 0;
}
