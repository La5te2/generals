// Build-time executable-icon generator; shares the window icon artwork through icon.hpp.
#include "icon.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>

namespace {
    // ICO/DIB fields use little-endian integers independently of the machine running this build helper.
    void integer(std::ostream& out, std::uint32_t value, int bytes = 4) {
        for (int i = 0; i < bytes; ++i) out.put(static_cast<char>((value >> (8 * i)) & 255));
    }
    int maskSize(int size) { return ((size + 31) / 32) * 4 * size; }
    int imageSize(int size) { return 40 + size * size * 4 + maskSize(size); }
}

// Write all supported icon sizes to the single output path supplied by CMake; report usage or write failures with exit code 1.
int main(int argc, char** argv) try {
    if (argc != 2) { std::cerr << "Usage: icon <output.ico>\n"; return 1; }
    std::ofstream out(argv[1], std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    constexpr std::array sizes{16, 24, 32, 48, 64, 128, 256};
    integer(out, 0, 2); integer(out, 1, 2); integer(out, static_cast<std::uint32_t>(sizes.size()), 2);
    std::uint32_t offset = 6 + 16 * static_cast<std::uint32_t>(sizes.size());
    for (int size : sizes) {
        integer(out, size == 256 ? 0 : size, 1); integer(out, size == 256 ? 0 : size, 1);
        integer(out, 0, 2); integer(out, 1, 2); integer(out, 32, 2);
        integer(out, imageSize(size)); integer(out, offset);
        offset += imageSize(size);
    }
    for (int size : sizes) {
        // ICO bitmaps store bottom-up BGRA rows followed by a padded one-bit transparency mask.
        integer(out, 40); integer(out, size); integer(out, size * 2);
        integer(out, 1, 2); integer(out, 32, 2); integer(out, 0); integer(out, imageSize(size) - 40);
        for (int field = 0; field < 4; ++field) integer(out, 0);
        auto pixels = NEBULA::Icon::pixels(size);
        for (int y = size - 1; y >= 0; --y) for (int x = 0; x < size; ++x) {
            int pixel = (y * size + x) * 4;
            out.put(static_cast<char>(pixels[pixel + 2]));
            out.put(static_cast<char>(pixels[pixel + 1]));
            out.put(static_cast<char>(pixels[pixel]));
            out.put(static_cast<char>(pixels[pixel + 3]));
        }
        int stride = ((size + 31) / 32) * 4;
        std::vector<unsigned char> mask(maskSize(size));
        for (int y = 0; y < size; ++y) for (int x = 0; x < size; ++x)
            if (pixels[(y * size + x) * 4 + 3] == 0) mask[(size - 1 - y) * stride + x / 8] |= 0x80 >> (x % 8);
        out.write(reinterpret_cast<const char*>(mask.data()), static_cast<std::streamsize>(mask.size()));
    }
    out.close();
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
