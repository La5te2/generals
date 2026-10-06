// encode and decode the LZ-String byte representation used by replay files.
// adapted from LZ-String 1.4.4 by Pieroxy, distributed under WTFPL 2. see ATTRIBUTION.txt.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace NEBULA::LZ {
    constexpr std::size_t textLimit = 16 * 1024 * 1024;

    // codes store their low bit first. the resulting bits fill big-endian 16-bit words.
    inline std::string compress(std::string_view text) {
        if (text.size() > textLimit) throw std::runtime_error("Replay text exceeds 16 MiB");
        struct Word { unsigned code; bool literal; };
        std::unordered_map<std::string, Word> dictionary;
        unsigned next = 3, width = 2, remaining = 2;
        unsigned word = 0, used = 0;
        std::string bytes;
        auto bits = [&](unsigned value, unsigned count) {
            for (unsigned bit = 0; bit < count; ++bit) {
                word = (word << 1) | ((value >> bit) & 1);
                if (++used == 16) {
                    bytes += static_cast<char>(word >> 8);
                    bytes += static_cast<char>(word & 255);
                    word = used = 0;
                }
            }
        };
        auto enlarge = [&] {
            if (--remaining == 0) { remaining = 1u << width; ++width; }
        };
        auto emit = [&](const std::string& phrase) {
            auto& entry = dictionary.at(phrase);
            if (entry.literal) {
                bits(0, width);
                bits(static_cast<unsigned char>(phrase.front()), 8);
                entry.literal = false;
                enlarge();
            } else bits(entry.code, width);
            enlarge();
        };
        std::string phrase;
        // the caller supplies ASCII JSON with escaped Unicode. decoding also accepts raw UTF-16 literals.
        for (unsigned char character : text) {
            if (character > 127) throw std::runtime_error("Replay encoder expects ASCII JSON");
            std::string letter(1, static_cast<char>(character));
            if (!dictionary.contains(letter)) dictionary.emplace(letter, Word{next++, true});
            std::string joined = phrase + letter;
            if (dictionary.contains(joined)) phrase = std::move(joined);
            else {
                emit(phrase);
                dictionary.emplace(std::move(joined), Word{next++, false});
                phrase = std::move(letter);
            }
        }
        if (!phrase.empty()) emit(phrase);
        bits(2, width);
        bits(0, 16 - used);
        return bytes;
    }

    inline std::u16string decompress(std::string_view bytes) {
        if (bytes.empty() || bytes.size() % 2 != 0) throw std::runtime_error("Invalid replay compression");
        std::size_t cursor = 0;
        auto bits = [&](unsigned count) {
            if (cursor + count > bytes.size() * 8) throw std::runtime_error("Truncated replay compression");
            unsigned value = 0;
            for (unsigned bit = 0; bit < count; ++bit, ++cursor) {
                unsigned byte = static_cast<unsigned char>(bytes[cursor / 8]);
                value |= ((byte >> (7 - cursor % 8)) & 1) << bit;
            }
            return value;
        };
        auto finish = [&] {
            auto padding = 16 - cursor % 16;
            if (bytes.size() * 8 - cursor != padding || bits(static_cast<unsigned>(padding)) != 0) {
                throw std::runtime_error("Invalid replay compression ending");
            }
        };
        // dictionary phrases reference an earlier phrase plus one character, bounding memory per entry.
        struct Word { unsigned prefix; char16_t last, first; std::size_t length; };
        std::vector<Word> dictionary(3);
        auto literal = [&](unsigned count) {
            char16_t character = static_cast<char16_t>(bits(count));
            dictionary.push_back({0, character, character, 1});
            return static_cast<unsigned>(dictionary.size() - 1);
        };
        unsigned code = bits(2);
        if (code > 1) {
            if (code == 2) { finish(); return {}; }
            throw std::runtime_error("Invalid replay compression header");
        }
        unsigned previous = literal(code == 0 ? 8 : 16);
        unsigned width = 3, remaining = 4;
        std::u16string text(1, dictionary[previous].first);
        auto enlarge = [&] {
            if (--remaining == 0) { remaining = 1u << width; ++width; }
        };
        auto extend = [&](char16_t character) {
            const auto& parent = dictionary[previous];
            dictionary.push_back({previous, character, parent.first, parent.length + 1});
        };
        for (;;) {
            code = bits(width);
            if (code == 2) { finish(); return text; }
            if (code < 2) { code = literal(code == 0 ? 8 : 16); enlarge(); }
            if (code < 3 || code > dictionary.size()) throw std::runtime_error("Invalid replay dictionary code");
            bool self = code == dictionary.size();
            if (self) extend(dictionary[previous].first);
            auto length = dictionary[code].length;
            if (length > textLimit - text.size() || dictionary.size() > textLimit) {
                throw std::runtime_error("Decoded replay exceeds its size limit");
            }
            std::size_t end = text.size() + length;
            text.resize(end);
            for (unsigned phrase = code; phrase != 0; phrase = dictionary[phrase].prefix) {
                text[--end] = dictionary[phrase].last;
            }
            if (!self) extend(dictionary[code].first);
            enlarge();
            previous = code;
        }
    }
}
