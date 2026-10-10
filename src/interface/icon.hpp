// Rasterize the existing general outline for window icons and executable resources, without image assets.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace NEBULA::Icon {
    #include "icons.inc"

    // Return top-down RGBA pixels: a white crown on rounded black backing with transparent corners.
    inline std::vector<unsigned char> pixels(int size) {
        if (size < 1 || size > 256) throw std::invalid_argument("Icon size must be between 1 and 256");
        constexpr int samples = 4;
        int extent = size * samples;
        std::vector<unsigned char> mask(extent * extent);
        float width = static_cast<float>(crownBounds[2] - crownBounds[0]), height = static_cast<float>(crownBounds[3] - crownBounds[1]);
        float scale = extent * .84f / std::max(width, height);
        float left = (extent - width * scale) / 2 - crownBounds[0] * scale;
        float top = (extent - height * scale) / 2 - crownBounds[1] * scale;
        struct Point { float x, y; };
        auto point = [&](int x, int y) { return Point{left + x * scale, top + y * scale}; };
        auto edge = [](Point a, Point b, Point p) { return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x); };
        // Union triangles before downsampling so shared edges do not leave antialiasing seams.
        for (const auto& face : crownIcon) {
            Point a = point(face.ax, face.ay), b = point(face.bx, face.by), c = point(face.cx, face.cy);
            float area = edge(a, b, c);
            if (area == 0) continue;
            int x0 = std::max(0, static_cast<int>(std::floor(std::min({a.x, b.x, c.x}))));
            int y0 = std::max(0, static_cast<int>(std::floor(std::min({a.y, b.y, c.y}))));
            int x1 = std::min(extent, static_cast<int>(std::ceil(std::max({a.x, b.x, c.x}))));
            int y1 = std::min(extent, static_cast<int>(std::ceil(std::max({a.y, b.y, c.y}))));
            for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) {
                Point p{x + .5f, y + .5f};
                float ab = edge(a, b, p), bc = edge(b, c, p), ca = edge(c, a, p);
                if ((ab >= 0 && bc >= 0 && ca >= 0) || (ab <= 0 && bc <= 0 && ca <= 0)) mask[y * extent + x] = 255;
            }
        }
        std::vector<unsigned char> rgba(size * size * 4);
        float radius = extent * .20f;
        for (int y = 0; y < size; ++y) for (int x = 0; x < size; ++x) {
            int sum = 0, covered = 0;
            for (int dy = 0; dy < samples; ++dy) for (int dx = 0; dx < samples; ++dx) {
                float px = x * samples + dx + .5f, py = y * samples + dy + .5f;
                float rx = px - std::clamp(px, radius, extent - radius);
                float ry = py - std::clamp(py, radius, extent - radius);
                if (rx * rx + ry * ry <= radius * radius) {
                    ++covered;
                    sum += mask[(y * samples + dy) * extent + x * samples + dx];
                }
            }
            int offset = (y * size + x) * 4;
            rgba[offset] = rgba[offset + 1] = rgba[offset + 2] = static_cast<unsigned char>(covered ? sum / covered : 0);
            rgba[offset + 3] = static_cast<unsigned char>(255 * covered / (samples * samples));
        }
        return rgba;
    }
}
