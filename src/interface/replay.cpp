// local replay storage: save changed cells, then restore full snapshots during loading for direct playback access.
#include "replay.hpp"
#include <charconv>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

namespace NEBULA {
    namespace {
        constexpr std::size_t maxFileBytes = 512 * 1024 * 1024;
        constexpr int digestLength = 8;

        // FNV-1a identifies file contents and detects accidental changes. it is separate from game protocol fields.
        std::string digest(std::string_view bytes) {
            std::uint32_t value = 2166136261u;
            for (unsigned char byte : bytes) { value ^= byte; value *= 16777619u; }
            std::ostringstream output;
            output.imbue(std::locale::classic());
            output << std::hex << std::setfill('0') << std::setw(digestLength) << value;
            return output.str();
        }

        bool readFile(const std::filesystem::path& path, std::string& bytes, std::string& error) {
            std::error_code status;
            auto size = std::filesystem::file_size(path, status);
            if (status || size > maxFileBytes) { error = "Replay file is unavailable or exceeds 512 MiB"; return false; }
            std::ifstream input(path, std::ios::binary);
            bytes.resize(static_cast<std::size_t>(size));
            if (!input.read(bytes.data(), static_cast<std::streamsize>(size)) || input.peek() != EOF) {
                error = "Replay file could not be read completely";
                return false;
            }
            return true;
        }

        bool endLine(std::istream& input) {
            input >> std::ws;
            return input.eof();
        }

        std::string base36(std::int64_t value) {
            char buffer[32];
            auto converted = std::to_chars(buffer, buffer + sizeof(buffer), value, 36);
            return std::string(buffer, converted.ptr);
        }

        template<class Integer>
        bool number36(std::string_view text, Integer& value) {
            if (text.empty() || text.front() == '-') return false;
            auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, 36);
            return error == std::errc{} && end == text.data() + text.size();
        }

        std::string packCell(const Cell& cell) {
            std::string token;
            token += static_cast<char>('0' + static_cast<int>(cell.terrain));
            token += static_cast<char>('1' + cell.owner);
            return token + base36(cell.army);
        }

        std::optional<Cell> unpackCell(std::string_view token) {
            if (token.size() < 3 || token[0] < '1' || token[0] > '4' || token[1] < '0' || token[1] > '2') return std::nullopt;
            Cell cell;
            cell.terrain = static_cast<Terrain>(token[0] - '0');
            cell.owner = static_cast<std::int8_t>(token[1] - '1');
            if (!number36(token.substr(2), cell.army)) return std::nullopt;
            if (cell.terrain == Terrain::Mountain && (cell.owner != -1 || cell.army != 0)) return std::nullopt;
            return cell;
        }

        // read cells row by row. a token contains terrain (1..4), owner (0 neutral, 1 red, 2 blue), then army in base 36.
        // identical consecutive cells share a base-36 repeat count: "a*100.200.411" means ten empty plains,
        // one mountain and a red general with one soldier. each board string describes the entire board.
        std::string packBoard(const Board& board) {
            std::string packed;
            int begin = 0;
            while (begin < board.size()) {
                const auto& cell = board.at(begin / board.cols(), begin % board.cols());
                int end = begin + 1;
                while (end < board.size()) {
                    const auto& next = board.at(end / board.cols(), end % board.cols());
                    if (cell.terrain != next.terrain || cell.owner != next.owner || cell.army != next.army) break;
                    ++end;
                }
                if (!packed.empty()) packed += '.';
                if (end - begin > 1) packed += base36(end - begin) + '*';
                packed += packCell(cell);
                begin = end;
            }
            return packed;
        }

        // later file rows store only changed cells as "index:new-cell", separated by dots.
        // indices use row-major order and base 36. a single dash denotes an unchanged board.
        std::string changedCells(const Board& before, const Board& after) {
            std::string changes;
            for (int index = 0; index < after.size(); ++index) {
                const auto& old = before.at(index / after.cols(), index % after.cols());
                const auto& cell = after.at(index / after.cols(), index % after.cols());
                if (old.terrain == cell.terrain && old.owner == cell.owner && old.army == cell.army) continue;
                if (!changes.empty()) changes += '.';
                changes += base36(index) + ':' + packCell(cell);
            }
            return changes.empty() ? "-" : changes;
        }

        bool applyChanges(Board& board, std::string_view changes) {
            if (changes == "-") return true;
            if (changes.empty()) return false;
            int previous = -1;
            while (!changes.empty()) {
                auto end = changes.find('.');
                auto token = changes.substr(0, end);
                auto separator = token.find(':');
                int index;
                if (separator == std::string_view::npos || !number36(token.substr(0, separator), index) ||
                    index <= previous || index >= board.size()) return false;
                auto cell = unpackCell(token.substr(separator + 1));
                if (!cell) return false;
                board.at(index / board.cols(), index % board.cols()) = *cell;
                previous = index;
                if (end == std::string_view::npos) break;
                changes.remove_prefix(end + 1);
                if (changes.empty()) return false;
            }
            return true;
        }

        std::optional<States> unpackFrame(const ReplayFrame& frame, int rows, int cols) {
            if (frame.idle > frame.tick || static_cast<int>(frame.result) > 3) return std::nullopt;
            States state(rows, cols);
            state.tick = frame.tick;
            state.idle = frame.idle;
            state.result = frame.result;
            std::array<int, 2> generals{};
            std::array<std::int64_t, 2> totals{};
            std::string_view remaining = frame.board;
            int index = 0;
            while (!remaining.empty()) {
                auto delimiter = remaining.find('.');
                auto token = remaining.substr(0, delimiter);
                int repeat = 1;
                auto marker = token.find('*');
                if (marker != std::string_view::npos) {
                    if (!number36(token.substr(0, marker), repeat) || repeat < 2) return std::nullopt;
                    token.remove_prefix(marker + 1);
                }
                if (repeat > state.board.size() - index) return std::nullopt;
                auto decoded = unpackCell(token);
                if (!decoded) return std::nullopt;
                const auto& cell = *decoded;
                if (cell.terrain == Terrain::General) {
                    if (cell.owner < 0) return std::nullopt;
                    generals[cell.owner] += repeat;
                }
                if (cell.owner >= 0) {
                    auto& total = totals[cell.owner];
                    if (cell.army > (std::numeric_limits<std::int64_t>::max() - total) / repeat) return std::nullopt;
                    total += cell.army * repeat;
                }
                for (int count = 0; count < repeat; ++count, ++index) state.board.at(index / cols, index % cols) = cell;
                if (delimiter == std::string_view::npos) break;
                remaining.remove_prefix(delimiter + 1);
                if (remaining.empty()) return std::nullopt;
            }
            if (index != state.board.size() || generals[0] > 1 || generals[1] > 1) return std::nullopt;
            if (state.result == Phases::Ongoing && generals != std::array{1, 1}) return std::nullopt;
            return state;
        }

        bool readFrame(std::string_view line, ReplayFrame& frame) {
            auto first = line.find('|'), second = line.find('|', first == std::string_view::npos ? line.size() : first + 1);
            if (first == std::string_view::npos || second == std::string_view::npos) return false;
            std::istringstream position(std::string(line.substr(0, first)));
            int result;
            if (!Protocol::integer(position, frame.tick) || !Protocol::integer(position, frame.idle) ||
                !Protocol::integer(position, result) || result < 0 || result > 3 ||
                !(position >> frame.board) || !endLine(position)) return false;
            frame.result = static_cast<Phases>(result);
            std::istringstream red(std::string(line.substr(first + 1, second - first - 1)));
            std::istringstream blue(std::string(line.substr(second + 1)));
            auto redAction = Protocol::readAction(red), blueAction = Protocol::readAction(blue);
            if (!redAction || !blueAction) return false;
            frame.actions = {*redAction, *blueAction};
            return true;
        }
    }

    Recording::Recording(const States& initial) : rows(initial.board.rows()), cols(initial.board.cols()) {
        // frame zero holds the starting position. its two pass values stand for the absence of earlier moves.
        append(initial, {});
    }

    void Recording::append(const States& state, const std::array<Action, 2>& actions) {
        frames.push_back({packBoard(state.board), actions, state.tick, state.idle, state.result});
    }

    bool replayDirectory(const std::filesystem::path& directory, std::string& error) {
        error.clear();
        std::error_code status;
        std::filesystem::create_directories(directory, status);
        if (status || !std::filesystem::is_directory(directory, status)) {
            error = "Recording directory is unavailable";
            return false;
        }
        return true;
    }

    std::optional<std::filesystem::path> saveReplay(const Recording& record, const std::filesystem::path& directory,
                                                   std::string& error) {
        error = "Recording contains an invalid position or action";
        if (!Protocol::valid({0, record.rows, record.cols}) || record.frames.empty() || record.frames.size() > tickLimit + 3) {
            return std::nullopt;
        }
        std::ostringstream output;
        output.imbue(std::locale::classic());
        output << record.rows << ' ' << record.cols << ' ' << record.frames.size() << '\n';
        // the first row stores a full board. each later row stores the cells changed by that half-turn.
        // tick, idle count and result preserve the remaining state alongside the two submitted actions.
        // actions use the shared five-integer syntax and produced the position on the same line.
        std::optional<States> previous;
        for (std::size_t index = 0; index < record.frames.size(); ++index) {
            const auto& frame = record.frames[index];
            auto state = unpackFrame(frame, record.rows, record.cols);
            if (frame.tick != index || !state ||
                (index + 1 < record.frames.size() && frame.result != Phases::Ongoing)) return std::nullopt;
            std::string board = previous ? changedCells(previous->board, state->board) : frame.board;
            output << frame.tick << ' ' << frame.idle << ' ' << static_cast<int>(frame.result) << ' ' << board;
            for (const auto& action : frame.actions) {
                std::ostringstream encoded;
                if (!Protocol::writeAction(encoded, action)) return std::nullopt;
                std::string text = encoded.str();
                text.pop_back();
                output << " | " << text;
            }
            output << '\n';
            previous = std::move(state);
        }
        std::string bytes = output.str();
        if (bytes.size() + digestLength + 1 > maxFileBytes) { error = "Recording exceeds 512 MiB"; return std::nullopt; }
        std::string hash = digest(bytes);
        bytes += hash + '\n';
        if (!replayDirectory(directory, error)) return std::nullopt;
        auto path = directory / (hash + ".grf");
        std::error_code status;
        if (std::filesystem::exists(path, status)) {
            std::string existing;
            if (readFile(path, existing, error) && existing == bytes) return path;
            error = "Replay filename already contains different data";
            return std::nullopt;
        }
        // publish a completed file with rename. an interrupted write leaves the .part file separate from replays.
        auto temporary = directory / (hash + "." + std::to_string(std::random_device{}()) + ".part");
        std::ofstream file(temporary, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        file.flush();
        bool written = static_cast<bool>(file);
        file.close();
        written = written && !file.fail();
        if (written) std::filesystem::rename(temporary, path, status);
        if (!written || status) {
            std::error_code cleanup;
            std::filesystem::remove(temporary, cleanup);
            error = "Replay save failed. Check directory permissions and free space";
            return std::nullopt;
        }
        return path;
    }

    bool Replay::load(const std::filesystem::path& path, std::string& error) {
        error.clear();
        std::string bytes;
        if (!readFile(path, bytes, error)) return false;
        error = "Replay file is malformed";
        if (bytes.size() < digestLength + 2 || bytes.back() != '\n') return false;
        auto checksum = bytes.size() - digestLength - 1;
        if (bytes[checksum - 1] != '\n' || bytes.substr(checksum, digestLength) != digest(std::string_view(bytes).substr(0, checksum))) {
            error = "Replay checksum differs from its contents";
            return false;
        }
        std::istringstream input(bytes.substr(0, checksum));
        input.imbue(std::locale::classic());
        std::string line;
        std::getline(input, line);
        std::istringstream header(line);
        int rows, cols;
        std::size_t count;
        if (!Protocol::integer(header, rows) || !Protocol::integer(header, cols) || !Protocol::valid({0, rows, cols}) ||
            !Protocol::integer(header, count) || count == 0 || count > tickLimit + 3 || !endLine(header)) return false;
        Replay next;
        std::optional<States> previous;
        for (std::size_t index = 0; index < count; ++index) {
            ReplayFrame frame;
            if (!std::getline(input, line) || !readFrame(line, frame) || frame.tick != index ||
                (index + 1 < count && frame.result != Phases::Ongoing)) return false;
            // expand file differences once. seeking then reads independent snapshots, with no rule execution.
            if (index > 0) {
                if (!applyChanges(previous->board, frame.board)) return false;
                frame.board = packBoard(previous->board);
            }
            auto state = unpackFrame(frame, rows, cols);
            if (!state) return false;
            if (index == 0) next.position = state;
            previous = std::move(state);
            next.frames.push_back(std::move(frame));
        }
        if (!endLine(input)) return false;
        *this = std::move(next);
        error.clear();
        return true;
    }

    bool Replay::seek(std::size_t halfTurn) {
        if (!position || halfTurn >= frames.size()) return false;
        auto next = unpackFrame(frames[halfTurn], position->board.rows(), position->board.cols());
        if (!next) return false;
        position = std::move(next);
        current = halfTurn;
        return true;
    }
}
