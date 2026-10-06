// replay files contain an initial map and timed events. playback reconstructs positions with the rule engine.
#include "replay.hpp"
#include "lz.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

namespace NEBULA {
    namespace {
        using Json = nlohmann::json;
        constexpr std::size_t maxFileBytes = 16 * 1024 * 1024;
        constexpr std::size_t maxTurns = tickLimit + 2;

        // array positions belong to the .gior file format, independently of the agent action protocol.
        enum Field : std::size_t {
            Format, Identity, Width, Height, Players, Stars, Cities, CityArmies, Generals, Mountains,
            Moves, Afks, Teams, Map, Neutrals, NeutralArmies, Swamps, Chat, Colors, Lights, Settings,
            Modifiers, Observatories, Lookouts, Deserts, Transforms, Pings, Trades, Tunnels, TunnelLimits,
            ClockTimes, StrongholdDensity, StrongholdMinimum, StrongholdMaximum, Strongholds,
            StrongholdArmies, ClockTimestamps, ModifierOptions, FieldCount
        };

        void require(bool valid, const char* message) {
            if (!valid) throw std::runtime_error(message);
        }

        std::int64_t integer(const Json& value, std::int64_t low, std::int64_t high) {
            require(value.is_number_integer(), "Replay integer field is malformed");
            if (value.is_number_unsigned()) require(value.get<std::uint64_t>() <= static_cast<std::uint64_t>(high),
                                                    "Replay integer exceeds its range");
            auto result = value.get<std::int64_t>();
            require(result >= low && result <= high, "Replay integer exceeds its range");
            return result;
        }

        const Json& field(const Json& data, Field index) {
            static const Json empty = Json::array();
            return index < data.size() && !data[index].is_null() ? data[index] : empty;
        }

        std::string digest(std::string_view bytes) {
            std::uint32_t value = 2166136261u;
            for (unsigned char byte : bytes) { value ^= byte; value *= 16777619u; }
            std::ostringstream output;
            output.imbue(std::locale::classic());
            output << std::hex << std::setfill('0') << std::setw(8) << value;
            return output.str();
        }

        bool readFile(const std::filesystem::path& path, std::string& bytes, std::string& error) {
            std::error_code status;
            auto size = std::filesystem::file_size(path, status);
            if (status || size > maxFileBytes) { error = "Replay file is unavailable or exceeds 16 MiB"; return false; }
            std::ifstream input(path, std::ios::binary);
            bytes.resize(static_cast<std::size_t>(size));
            if (!input.read(bytes.data(), static_cast<std::streamsize>(size)) || input.peek() != EOF) {
                error = "Replay file could not be read completely";
                return false;
            }
            return true;
        }

        States initialPosition(const Json& data) {
            require(data.is_array() && data.size() >= Chat + 1, "Expected a .gior replay array");
            // older files use different move priorities. future revisions also require a schema review.
            require(integer(data[Format], 0, 1000) >= 15 && data[Format] <= 19,
                    "Replay requires a different rule revision (supported: 15 through 19)");
            require(data[Players].is_array() && data[Players].size() == 2 &&
                    data[Generals].is_array() && data[Generals].size() == 2, "Replay requires exactly two players");
            for (Field index : {Swamps, Lights, Modifiers, Observatories, Lookouts, Deserts, Tunnels, Strongholds}) {
                require(field(data, index).is_array() && field(data, index).empty(),
                        "Replay uses terrain or modifiers outside mainstream 1v1");
            }
            const auto& teams = field(data, Teams);
            require(teams.is_array() && (teams.empty() || (teams.size() == 2 && teams[0].is_number_integer() &&
                    teams[1].is_number_integer() && teams[0] != teams[1])), "Replay requires opposing players");
            States state(static_cast<int>(integer(data[Height], 1, dim)), static_cast<int>(integer(data[Width], 1, dim)));
            auto cellAt = [&](const Json& index) -> Cell& {
                int value = static_cast<int>(integer(index, 0, state.board.size() - 1));
                return state.board.at(value / state.board.cols(), value % state.board.cols());
            };
            auto place = [&](Field indices, Terrain terrain, Field armies) {
                const auto& cells = field(data, indices);
                require(cells.is_array() && cells.size() <= static_cast<std::size_t>(state.board.size()), "Invalid replay map indices");
                if (armies != FieldCount) require(field(data, armies).is_array() && field(data, armies).size() == cells.size(),
                                                  "Replay map armies differ from map indices");
                for (std::size_t index = 0; index < cells.size(); ++index) {
                    auto& cell = cellAt(cells[index]);
                    require(cell.terrain == Terrain::Plain && cell.owner == -1, "Replay terrain overlaps");
                    cell.terrain = terrain;
                    if (armies != FieldCount) cell.army = integer(data[armies][index], 0,
                        std::numeric_limits<std::int64_t>::max() / Capacity - 2 * maxTurns);
                }
            };
            place(Mountains, Terrain::Mountain, FieldCount);
            place(Cities, Terrain::City, CityArmies);
            for (int player = 0; player < 2; ++player) {
                auto& cell = cellAt(data[Generals][player]);
                require(cell.terrain == Terrain::Plain && cell.owner == -1, "Invalid starting general");
                cell = {1, static_cast<std::int8_t>(player), Terrain::General};
            }
            int red = data[Generals][0].get<int>(), blue = data[Generals][1].get<int>();
            require(std::abs(red / state.board.cols() - blue / state.board.cols()) +
                    std::abs(red % state.board.cols() - blue % state.board.cols()) >= 3,
                    "Replay general spacing exceeds supported map rules");
            const auto& neutrals = field(data, Neutrals);
            const auto& armies = field(data, NeutralArmies);
            require(neutrals.is_array() && armies.is_array() && neutrals.size() == armies.size() &&
                    neutrals.size() <= static_cast<std::size_t>(state.board.size()), "Invalid neutral army data");
            // these entries override initial armies, including custom starting armies on generals.
            for (std::size_t index = 0; index < neutrals.size(); ++index) {
                auto& cell = cellAt(neutrals[index]);
                require(cell.terrain != Terrain::Mountain, "Replay places an army on a mountain");
                cell.army = integer(armies[index], 0, std::numeric_limits<std::int64_t>::max() / Capacity - 2 * maxTurns);
            }
            return state;
        }

        Json encode(const Recording& record) {
            const auto& state = record.initial;
            require(state.tick == 0 && state.idle == 0 && state.result == Phases::Ongoing, "Recording must start at initial setup");
            Json data = std::vector<Json>(FieldCount, Json::array());
            data[Format] = 19;
            data[Identity] = "local";
            data[Width] = state.board.cols();
            data[Height] = state.board.rows();
            data[Players] = {"RED", "BLUE"};
            data[Stars] = {0, 0};
            data[Generals] = {nullptr, nullptr};
            data[Teams] = data[Map] = data[ModifierOptions] = nullptr;
            data[Colors] = {0, 1};
            data[Transforms] = {0, 0};
            data[Settings] = {1, 0.5, 0.5, 0, 0.5, 0.5, 0, 0, 0};
            data[StrongholdDensity] = 0;
            data[StrongholdMinimum] = 10;
            data[StrongholdMaximum] = 25;
            for (int index = 0; index < state.board.size(); ++index) {
                const auto& cell = state.board.at(index / state.board.cols(), index % state.board.cols());
                require(cell.army >= 0 && cell.owner >= -1 && cell.owner <= 1, "Invalid recording setup");
                if (cell.terrain == Terrain::General) {
                    require(cell.owner >= 0 && data[Generals][cell.owner].is_null(), "Invalid recording generals");
                    data[Generals][cell.owner] = index;
                } else {
                    require(cell.owner == -1, ".gior initial ownership belongs to generals");
                    if (cell.terrain == Terrain::Mountain) {
                        require(cell.army == 0, "Mountain contains an army");
                        data[Mountains].push_back(index);
                    } else if (cell.terrain == Terrain::City) {
                        data[Cities].push_back(index);
                        data[CityArmies].push_back(cell.army);
                    } else require(cell.terrain == Terrain::Plain, "Invalid recording terrain");
                }
                if ((cell.terrain == Terrain::Plain && cell.army > 0) || (cell.terrain == Terrain::General && cell.army != 1)) {
                    data[Neutrals].push_back(index);
                    data[NeutralArmies].push_back(cell.army);
                }
            }
            initialPosition(data);
            for (const auto& move : record.moves) {
                const auto& action = move.action;
                int from = action.row * state.board.cols() + action.col;
                int to = from;
                switch (action.direction) {
                    case Direction::Up: to -= state.board.cols(); break;
                    case Direction::Down: to += state.board.cols(); break;
                    case Direction::Left: --to; break;
                    case Direction::Right: ++to; break;
                }
                data[Moves].push_back({move.player, from, to, int(action.half), move.tick});
            }
            if (record.surrendered) {
                require(record.previous.tick > 0, "Surrender requires a completed half-turn");
                data[Afks].push_back({*record.surrendered, record.previous.tick - 1});
            }
            return data;
        }
    }

    void Recording::append(const States& state, const std::array<Action, 2>& actions) {
        for (int player = 0; player < 2; ++player) {
            if (actions[player].type == ActionType::Move && legit(previous, player, actions[player])) {
                moves.push_back({previous.tick, player, actions[player]});
            }
        }
        previous = state;
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
        error.clear();
        std::string bytes;
        try { bytes = LZ::compress(encode(record).dump(-1, ' ', true)); }
        catch (const std::exception& problem) { error = problem.what(); return std::nullopt; }
        std::string hash = digest(bytes);
        if (!replayDirectory(directory, error)) return std::nullopt;
        auto path = directory / (hash + ".gior");
        std::error_code status;
        // compare complete contents before reusing a short digest filename.
        if (std::filesystem::exists(path, status)) {
            std::string existing;
            if (readFile(path, existing, error) && existing == bytes) return path;
            error = "Replay filename already contains different data";
            return std::nullopt;
        }
        // rename publishes the completed recording. interrupted writes remain separate .part files.
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

    void Replay::advance(States& state) const {
        const auto& turn = turns[static_cast<std::size_t>(state.tick)];
        auto actions = turn.actions;
        for (int player = 0; player < 2; ++player) if (turn.surrendered[player]) actions[player] = {};
        ::step(state, actions);
        // surrender ends participation while preserving land. the same half-turn still completes its growth.
        if (turn.surrendered[0] || turn.surrendered[1]) {
            state.result = turn.surrendered[0] ? Phases::BlueWin : Phases::RedWin;
        }
    }

    bool Replay::load(const std::filesystem::path& path, std::string& error) {
        error.clear();
        std::string bytes;
        if (!readFile(path, bytes, error)) return false;
        try {
            auto text = LZ::decompress(bytes);
            auto data = Json::parse(text, [](int depth, Json::parse_event_t, Json&) {
                require(depth <= 32, "Replay JSON nesting exceeds its limit");
                return true;
            });
            States state = initialPosition(data);
            Replay next;
            next.position = state;
            for (int player = 0; player < 2; ++player) {
                require(data[Players][player].is_string(), "Invalid replay player name");
                next.players[player] = data[Players][player].get<std::string>();
                require(next.players[player].size() <= 256, "Replay player name is too long");
            }
            next.turns.resize(maxTurns);
            const auto& moves = field(data, Moves);
            require(moves.is_array() && moves.size() <= 2 * maxTurns, "Invalid replay move list");
            std::size_t previousTick = 0;
            for (const auto& move : moves) {
                require(move.is_array() && move.size() == 5, "Replay move needs five integer fields");
                int player = static_cast<int>(integer(move[0], 0, 1));
                int from = static_cast<int>(integer(move[1], 0, state.board.size() - 1));
                int to = static_cast<int>(integer(move[2], 0, state.board.size() - 1));
                bool half = integer(move[3], 0, 1) != 0;
                auto tick = static_cast<std::size_t>(integer(move[4], previousTick, maxTurns - 1));
                previousTick = tick;
                int row = from / state.board.cols(), col = from % state.board.cols();
                int dr = to / state.board.cols() - row, dc = to % state.board.cols() - col;
                require(std::abs(dr) + std::abs(dc) == 1, "Replay move requires adjacent cells");
                auto& action = next.turns[tick].actions[player];
                require(action.type == ActionType::Pass, "Replay contains duplicate moves in a half-turn");
                action = {ActionType::Move, row, col,
                    dr < 0 ? Direction::Up : dr > 0 ? Direction::Down : dc < 0 ? Direction::Left : Direction::Right, half};
            }
            const auto& afks = field(data, Afks);
            require(afks.is_array() && afks.size() <= 4, "Invalid replay surrender list");
            previousTick = 0;
            for (const auto& event : afks) {
                require(event.is_array() && event.size() == 2, "Invalid replay surrender event");
                int player = static_cast<int>(integer(event[0], 0, 1));
                auto tick = static_cast<std::size_t>(integer(event[1], previousTick, maxTurns - 1));
                previousTick = tick;
                require(!next.turns[tick].surrendered[0] && !next.turns[tick].surrendered[1],
                        "Replay contains multiple surrender events in one half-turn");
                next.turns[tick].surrendered[player] = true;
            }
            // one validation pass also builds checkpoints. seeking repeats at most 128 rule steps.
            while (state.result == Phases::Ongoing && state.tick < next.turns.size()) {
                if (state.tick % checkpointInterval == 0) next.checkpoints.push_back(state);
                for (int player = 0; player < 2; ++player) {
                    require(legit(state, player, next.turns[static_cast<std::size_t>(state.tick)].actions[player]),
                            "Replay move differs from the reconstructed position");
                }
                next.advance(state);
            }
            require(moves.empty() || moves.back()[4].get<std::uint64_t>() < state.tick,
                    "Replay contains moves after the reconstructed ending");
            next.turns.resize(static_cast<std::size_t>(state.tick));
            require(!next.checkpoints.empty(), "Replay contains an empty timeline");
            *this = std::move(next);
            return true;
        } catch (const Json::exception&) { error = "Replay JSON is malformed"; }
        catch (const std::exception& problem) { error = problem.what(); }
        return false;
    }

    bool Replay::seek(std::size_t halfTurn) {
        if (!position || halfTurn > turns.size()) return false;
        std::size_t checkpoint = std::min(halfTurn / checkpointInterval, checkpoints.size() - 1);
        const auto& saved = checkpoints[checkpoint];
        if (position->tick > halfTurn || position->tick < saved.tick) position = saved;
        while (position->tick < halfTurn) advance(*position);
        return true;
    }
}
