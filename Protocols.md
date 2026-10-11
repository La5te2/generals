# Protocols

This document describes how external agent programs exchange observations and actions with the interface, including message formats, reply timing, and session termination. It also specifies the GIOR replay file format.

## External Agents

An external agent runs as a separate program. To use it in the graphical interface, specify the program in that player's **Player** field. For each game, the interface starts one program per external player and connects its standard input and output to the game through pipes.

From the agent's perspective, initialization and observations are input, and actions are output. The agent reads game data from `stdin`, computes a decision, and writes its reply to `stdout`. This exchange applies to agents implemented in any language. An integration layer can use `engine/protocol.hpp` to convert these messages to and from `Observation` and `Action`, while delegating decisions to the agent's implementation.

### Streams and Lifecycle

The interface supplies the agent's input automatically through `stdin` and reads its replies from `stdout`. The agent writes logs, loading messages, and other diagnostics to `stderr`, keeping `stdout` dedicated to action lines.

Messages contain decimal integers separated by spaces, with each line terminated by LF. CRLF is also accepted. The line layouts below specify how many integers belong on each line.

The exchange begins with one initialization line. The interface then sends a complete observation, and the agent responds with one action line. After writing the line, the agent flushes `stdout` so that the interface can receive it immediately, then reads the next observation. This observation/reply cycle continues until EOF on `stdin` signals session termination.

The interface obtains the player's identity from that program's initialization and associates each reply with its preceding observation. The reply therefore contains the five action fields defined under **Actions**.

### Initialization

The first line read from the agent's `stdin` contains three decimal integers:

```text
player rows cols
```

`player` is the agent's seat, with `0` for red and `1` for blue. Let $H$ denote `rows` and $W$ denote `cols`. Both dimensions remain fixed throughout the game and satisfy

$$
1 \le H \le 40, \qquad 1 \le W \le 40.
$$

For example, `1 20 23` assigns the agent to blue on a board with 20 rows and 23 columns.

### Observations

After initialization, the agent reads each observation from `stdin`. An observation contains one statistics line followed by three matrices, for a total of $1 + 3H$ lines. Each matrix has $H$ rows of exactly $W$ integers, ordered from top to bottom and left to right.

```text
tick my_land my_army opp_land opp_army
<H rows of terrain codes>
<H rows of ownership codes>
<H rows of army counts>
```

The angle-bracketed lines above stand for matrix rows. In an actual message, they are replaced by integers. The agent reads all terrain rows first, then all ownership rows, and finally all army rows. Initialization fixes the dimensions, so reading these $1 + 3H$ lines completes one observation.

`tick` is the half-turn counter associated with the observation. LOCAL and LAN begin at `0`; main- and bot-server games retain the server's counter, which starts at `1`. The first observation delivered to an agent may have a later tick, so the agent uses the received value for game timing.

`my_land` and `my_army` give the agent's total land and army. `opp_land` and `opp_army` give the opponent's public totals. These statistics cover each player's entire territory, while the matrices describe the agent's current field of view. Land counts satisfy $0 \le L \le HW$, army totals satisfy $0 \le A \le 2^{63}-1$, and `tick` is an unsigned 64-bit integer.

Terrain codes are:

- `0`: fog.
- `1`: visible plain.
- `2`: visible mountain.
- `3`: visible city.
- `4`: visible general.
- `5`: a hidden obstacle, representing a mountain or city under fog.

Ownership is relative to the receiving agent: `0` means neutral or hidden, `1` means self, and `2` means opponent. Terrain codes distinguish visible neutral cells from hidden cells.

Army counts use signed 64-bit integers in the range $0 \le A \le 2^{63}-1$. Cells with terrain code `0` or `5` carry zero ownership and army values as placeholders for hidden information. On visible cells, an army value of `0` represents an actual empty garrison.

For example, after initialization `0 3 3`, the following observation describes the red agent's view of a three-by-three board. Its general is at `(0, 0)` with five soldiers, and its other cell is at `(1, 0)` with one soldier. An enemy cell at `(1, 1)` contains four soldiers. The opponent's public totals are three tiles and twelve soldiers; the remaining enemy cells are hidden. Within this message, the first line contains statistics, the next three lines contain terrain, the next three contain ownership, and the final three contain armies.

```text
8 2 6 3 12
4 1 5
1 1 0
1 1 0
1 0 0
1 2 0
0 0 0
5 0 0
1 4 0
0 0 0
```

### Actions

After reading a complete observation, the agent writes exactly five decimal integers to `stdout`, separated by spaces and followed by a newline:

```text
kind row col direction split
```

`kind` selects `0` for Move or `1` for Pass. For a move, `row` and `col` identify the source cell, with $0 \le \text{row} < H$ and $0 \le \text{col} < W$. Coordinates start at `(0, 0)` in the top-left corner, with rows increasing downward and columns increasing rightward.

`direction` selects `0` for up, `1` for down, `2` for left, or `3` for right. `split` selects `0` for a full move or `1` for a half move. Given source army $A \ge 2$ at execution time and split flag $s$, the number sent is

$$
S(A, s) =
\begin{cases}
A - 1 & s = 0, \\
\left\lfloor A/2 \right\rfloor & s = 1.
\end{cases}
$$

For example, a source with nine soldiers sends eight on a full move or four on a half move. The standard Pass line is `1 0 0 0 0`. The final four integers are placeholders for Pass.

For the observation above, a full move down from `(0, 0)` is written as

```text
0 0 0 1 0
```

A half move right from the same cell is written as

```text
0 0 0 3 1
```

These are alternative replies to one observation. The agent chooses one, writes that line, and flushes its output. With the C++ helper, constructing the corresponding `Action` and calling `Protocol::writeAction(std::cout, action)` produces the same text. The **Protocol Integration** example at the end of this section shows the complete exchange.

The direction determines the adjacent destination: up changes the row by `-1`, down by `+1`, left changes the column by `-1`, and right by `+1`. At execution time, a legal move starts from an owned cell with at least two soldiers and targets a cell inside the board whose terrain is a plain, city, or general. Its success in capturing that cell depends on the armies present when it executes.

### Timing and Observation Gaps

The game may advance while the agent is calculating. The interface waits for that observation's reply before sending another observation to the agent. If several half-turns have elapsed, the next observation is the latest available one. Consequently, consecutive received ticks may skip half-turns. Each observation contains complete current matrices; the agent maintains any history it needs.

If calculation outlasts a half-turn, the agent still completes that request by sending one action, including Pass if appropriate. This completes the exchange for the preceding observation and allows the interface to send the latest one. The reply to the older tick expires.

In LOCAL, each half-turn has a settlement deadline. An action participates only if its observation tick matches the half-turn being settled and its complete reply arrived by that deadline. An absent or late reply becomes Pass for that half-turn. After a missed deadline, the agent session continues with the same request/reply sequence. The interval follows the configured LOCAL half-turn duration.

In LAN, the game advances every 500 ms. Agent replies are eligible while their observation tick remains current; a move that reaches the receiving peer after the tick changes is discarded. Main- and bot-server games advance on server board updates and also discard replies to older observation ticks. Pass completes the response to that observation and contributes a passing action for that player.

The game clock continues during calculation. The agent should complete resource loading promptly and send each decision as soon as it is ready.

### Action Outcomes and Session End

Execution depends on timely acceptance and move legality at execution time. An action may expire before acceptance or become illegal before execution. The agent determines action outcomes from subsequent observations of the position.

An action line must contain exactly five integers and may occupy at most 255 bytes before LF. With CRLF, CR counts toward that limit. A malformed or oversized reply, a broken stream, or premature agent exit ends the agent session. A correctly formatted but illegal move leaves that player's action ineffective while the agent session continues. In LAN, agent failure also surrenders that player's side.

Observations describe ongoing positions. When the session ends, the interface closes the input stream; EOF is the agent's session-end notification. The agent handles EOF and exits. A program that remains blocked can be terminated.

### Protocol Integration

The C++ integration layer includes `engine/protocol.hpp`, which supplies the `Observation` and `Action` definitions and inline serialization functions. Passing `std::cin` and `std::cout` to those functions connects the integration layer to the streams established by the interface.

`Protocol::readInit(input)` reads initialization. `Protocol::readObservation(input, init)` reads a complete observation using its fixed dimensions. Both return `std::nullopt` on invalid input, EOF, or a read failure. `Protocol::writeAction(output, action)` writes one action line and reports output success; the caller must flush the stream afterward.

The protocol reader converts relative ownership codes in the text stream to absolute player indices in `Observation`:

- `view.player` is `0` for red or `1` for blue.
- `view.cells[row * view.cols + col].owner` is `-1` for neutral or hidden, `0` for red, or `1` for blue. A cell belongs to the agent when `cell.owner == view.player`.
- `view.land[player]` and `view.armies[player]` use the same red/blue indices. The agent's totals are at `view.player`; the opponent's are at `1 - view.player`.
- Terrain values are represented by `ViewTerrain`. Only the first `view.rows * view.cols` cells belong to the board.

An `Action` contains `type`, source `row` and `col`, `direction`, and the boolean `half`. Default construction produces Pass. For example, this constructs the half move right from `(0, 0)`:

```cpp
Action action{ActionType::Move, 0, 0, Direction::Right, true};
```

`Protocol::writeAction(std::cout, action)` serializes that object as `0 0 0 3 1` followed by a newline. Calling `std::cout.flush()` then sends the reply through the pipe.

An agent may also use the public engine for local simulation by including `engine/engine.hpp`. `Engine::reset(rows, cols, seed)` creates a game and returns whether initialization succeeded. After a successful reset, `Engine::observe(player)` returns an `std::optional<Observation>` for that player, and `Engine::step(actions)` resolves one half-turn using `actions[0]` for red and `actions[1]` for blue. Simulation runs inside the agent's program; its decisions are returned to the interface through the same input/output loop.

The following pseudocode shows this integration layer around an abstract `Agent`. `Agent` and its methods represent the author's agent implementation or runtime binding, which may invoke C++, Python, or another language. `initialize` supplies the seat and board dimensions; `decide` passes an observation to that implementation and returns its decision as an `Action`. The `Protocol` calls use the public API directly:

```cpp
#include "engine/protocol.hpp"
#include <iostream>

int main() {
    Agent agent; // Abstract agent implementation or runtime binding.

    // The interface writes "player rows cols\n" to the agent's stdin pipe.
    // readInit consumes that line from std::cin. For "0 3 3\n", init holds
    // player = 0 (red), rows = 3, and cols = 3.
    auto init = Protocol::readInit(std::cin);
    if (!init) return 1;
    agent.initialize(*init); // Supply this game's seat and board dimensions.

    while (std::cin.peek() != std::char_traits<char>::eof()) {
        // The same stdin pipe next supplies "tick my_land my_army opp_land opp_army\n",
        // followed by rows terrain lines, rows ownership lines, and rows army lines.
        // Each matrix line contains cols integers. For init = {0, 3, 3}, this call
        // reads all 10 lines of the observation shown above and returns an Observation.
        auto view = Protocol::readObservation(std::cin, *init);
        if (!view) return 1;

        // Invoke the agent's decision implementation with the received observation.
        // The integration layer represents its decision as an Action.
        Action action = agent.decide(*view);
        // Write "kind row col direction split\n" to stdout for the interface to read.
        // Pass becomes "1 0 0 0 0\n"; a half move right from (0, 0)
        // would be Action{ActionType::Move, 0, 0, Direction::Right, true},
        // producing "0 0 0 3 1\n".
        if (!Protocol::writeAction(std::cout, action)) return 1;
        // Deliver the complete reply before waiting for the next observation.
        std::cout.flush();
        if (!std::cout) return 1;
    }
    // EOF means that the interface has ended this agent's game session.
    return std::cin.bad() ? 1 : 0;
}
```

## Replays

### File Structure

After decompression, a GIOR file contains a JSON array, denoted here by `data`. Each array position identifies a field. The first field, `data[0]`, stores the **GIOR file format version number**; for example, `data[0] = 19` identifies version `19`.

The interface accepts format versions `15` through `19` for games between two opposing players, using plains, mountains, cities, and generals. The complete version `19` layout contains 38 fields, listed below. A file may omit trailing fields while retaining at least the first 18 fields, from `data[0]` through `data[17]`. Optional list fields that are omitted or set to `null` have the same meaning as `[]`.

The numbers on the left identify positions in `data`. Each list-valued field contains a nested JSON array:

```text
0   GIOR format version
1   replay ID
2   width
3   height
4   player names
5   player stars
6   city indices
7   city armies
8   general indices, in player order
9   mountain indices
10  moves
11  surrender events
12  teams
13  map metadata
14  initial army override indices
15  initial army override values
16  swamps
17  chat
18  player colors
19  lights
20  game settings
21  modifiers
22  observatories
23  lookouts
24  deserts
25  player transforms
26  pings
27  general trades
28  tunnels
29  tunnel limits
30  clock times
31  stronghold density
32  stronghold minimum army
33  stronghold maximum army
34  stronghold indices
35  stronghold armies
36  clock timestamps
37  modifier options
```

Fields `2` and `3` specify the board width $W$ and height $H$, with $1 \le W,H \le 40$. Within this board, map locations are represented by **cell indices**. Cells are numbered row by row, beginning at the top-left corner. Thus, a cell at row $r$ and column $c$ has index $i$, where

$$
i = rW + c, \qquad 0 \le i < HW.
$$

Conversely, the row and column corresponding to a cell index are

$$
r = \left\lfloor i/W \right\rfloor, \qquad c = i \bmod W.
$$

Fields `4` and `8` associate each player with a name and a starting general cell: `data[4] = [redName, blueName]` and `data[8] = [redGeneral, blueGeneral]`. Each array therefore contains exactly two entries, and each name is a JSON string whose UTF-8 representation occupies at most 256 bytes. The two general cells start under their respective players' ownership, and their Manhattan distance is at least three. Field `12` describes team membership; for opposing players, it contains `null`, `[]`, or two distinct integer team IDs.

Together with the general cells in field `8`, the city cells in field `6` and mountain cells in field `9` define the initial terrain. Every listed position is a valid cell index, and each cell has one terrain assignment. The remaining cells are neutral plains; cities also start neutral.

Initial armies are specified separately from terrain. Each general starts with one soldier, and each plain starts with zero. For cities, fields `6` and `7` form a pair of equally sized arrays: the city at `data[6][j]` starts with `data[7][j]` soldiers. Fields `14` and `15` form another such pair, assigning `data[15][j]` soldiers to the cell at `data[14][j]`. These assignments replace the initial army counts, including those on generals, while preserving terrain and ownership. Every assigned army count is a nonnegative integer, and its cell must be a plain, city, or general.

Fields `10` and `11` then describe how play proceeds from this initial position: they contain moves and surrender events, respectively. Their record formats are specified under **Moves and Endings** below.

The remaining fields carry additional terrain, game settings, and replay metadata. Their values are described below, followed by examples of a minimal replay and a replay with populated map and metadata fields.

### List Fields and Metadata

A list field uses `[]` when it contains zero entries. For a populated list, the element format depends on the field: terrain lists contain cell indices, paired army lists contain counts, and record lists contain nested arrays. The examples below use a board with width `4` and height `3`, whose cell indices are

```text
 0  1  2  3
 4  5  6  7
 8  9 10 11
```

For the initial map, list entries are written as follows:

```javascript
data[6] = [2];             // one city, at row 0, column 2
data[7] = [4];             // that city begins with 4 soldiers
data[9] = [5, 6];          // mountains at row 1, columns 1 and 2
data[14] = [0, 4, 11];     // cells with custom initial armies
data[15] = [12, 2, 8];     // 12 on cell 0, 2 on cell 4, 8 on cell 11
```

The positions in a paired list correspond by order. For example, the third entry of field `15` assigns eight soldiers to the third cell listed in field `14`, which is cell `11`. With generals at `[0, 11]`, this gives red twelve starting soldiers, blue eight, and the neutral plain at cell `4` two.

Fields associated with individual players also use player order. Field `5` stores their star ratings, for example `[12, 8]`; `null` may represent an unavailable rating. Field `18` stores player color IDs, with `[0, 1]` selecting the usual red and blue colors. Field `25` stores per-player view-transform codes, with `[0, 0]` selecting the original board orientation for both. Moves and map cells use the board's original cell indices.

Chat and map-marker lists contain records rather than single values:

```javascript
data[17] = [
    ["Ready.", "", 0, 0],       // text, display prefix, player, tick
    ["Good game.", "", 1, 3]
];
data[26] = [
    [0, 1, 2],                 // player, tick, cell index: red marks cell 2 at tick 1
    [1, 2, 10]                 // blue marks cell 10 at tick 2
];
```

Each chat record has the form `[text, prefix, player, tick]`, where `text` and `prefix` are strings. An empty prefix is `""`. Each map-marker record has the form `[player, tick, cell]`. In both lists, `player` is `0` for red or `1` for blue, and `tick` locates the record on the replay timeline. A record of a general trade in field `27` has the form `[playerA, playerB, tick]`; for example, `[[0, 1, 20]]` records a trade between the two players at tick `20`. These records describe occurrences in the recorded game.

Field `20` contains nine game-setting values in this order:

```text
[speed, cityDensity, mountainDensity, swampDensity,
 cityFairness, spawnFairness, desertDensity, lookoutDensity, observatoryDensity]
```

The examples use `[1, 0.5, 0.5, 0, 0.5, 0.5, 0, 0, 0]`. The explicit terrain lists specify the recorded map; density and fairness values describe its generation settings. Field `13` can contain the custom map's name as a string, such as `"example-map"`, or `null`.

For ordinary two-player recordings, the special-terrain and modifier lists in fields `16`, `19`, `21`, `22`, `23`, `24`, `28`, and `34` are required to be empty. Their paired lists, tunnel limits (`29`) and stronghold armies (`35`), therefore use `[]` as well. Fields `31`, `32`, and `33` use `0`, `10`, and `25`, respectively, and modifier options in field `37` use `null`.

Clock metadata uses seconds. Field `30`, when populated, contains one list per player, with each list recording remaining time at successive ticks, such as `[[60, 59.5], [60, 60]]`. Field `36` contains elapsed times of half-turn execution, such as `[0, 0.5]`. The examples use `[]` for both fields because they record ordinary fixed-interval games.

### File Encoding

The `.gior` file contains the bytes returned by LZ-String's `compressToUint8Array` applied to the JSON text of the array described above. Compression uses UTF-16 code units, and each compressed 16-bit word is stored as two bytes, high byte first. JSON strings may contain Unicode characters directly or represent them with JSON Unicode escapes.

The interface accepts `.gior` files up to 16 MiB in size.

### Moves and Endings

Field `10` contains the replay's move records. Each record is an array of five integers:

```text
[player, from, to, half, tick]
```

`player` selects `0` for red or `1` for blue. `from` and `to` are the cell indices of the source and destination, which must be orthogonally adjacent. `half` selects `0` for a full move or `1` for a half move. `tick` is the half-turn at which the move executes, with `0` identifying the initial position before any action.

The list is ordered by tick, with at most one move per player per half-turn. Both players may have a record at the same tick. A player passes at any tick for which that player has no move record. Thus, a tick with a record for one player includes a Pass for the other, and a tick with no records includes two Passes. Each recorded move must be legal in the position at its tick, and all records must precede the game's end.

Field `11` contains surrender records, each in the form `[player, tick]`, using the same player indices and tick convention as moves. At that tick, the surrendering player's action is Pass; the half-turn includes the other player's action and scheduled growth, and the other player wins. Surrender records are ordered by tick, with at most one surrender per half-turn. General captures and game-rule limits also determine termination, so a replay may end with an empty surrender list.

### Minimal Valid Example

This example uses GIOR format version `19` on a board with one row and four columns. Red's general occupies cell `0`, and blue's general occupies cell `3`. Both players pass at ticks `0` and `1`. At tick `2`, red moves its full available army from cell `0` to cell `1`. Blue surrenders at tick `3`, and red wins.

With the Node.js package `lz-string` installed, the following program writes `example.gior`:

```javascript
const { writeFileSync } = require("node:fs");
const { compressToUint8Array } = require("lz-string");

const data = [
    19,                           // 0: GIOR format version
    "example",                    // 1: replay ID
    4,                            // 2: width
    1,                            // 3: height
    ["RED", "BLUE"],              // 4: player names
    [0, 0],                       // 5: player stars
    [],                           // 6: city indices (empty)
    [],                           // 7: city armies (empty)
    [0, 3],                       // 8: red and blue general indices
    [],                           // 9: mountain indices (empty)
    [[0, 0, 1, 0, 2]],             // 10: red moves from 0 to 1, full army, tick 2
    [[1, 3]],                     // 11: blue surrenders at tick 3
    null,                         // 12: teams
    null,                         // 13: map metadata
    [],                           // 14: initial army override indices (empty)
    [],                           // 15: initial army override values (empty)
    [],                           // 16: swamps (empty)
    [],                           // 17: chat (empty)
    [0, 1],                       // 18: red and blue player colors
    [],                           // 19: lights (empty)
    [1, 0.5, 0.5, 0, 0.5, 0.5, 0, 0, 0], // 20: game settings
    [],                           // 21: modifiers (empty)
    [],                           // 22: observatories (empty)
    [],                           // 23: lookouts (empty)
    [],                           // 24: deserts (empty)
    [0, 0],                       // 25: player transforms
    [],                           // 26: pings (empty)
    [],                           // 27: general trades (empty)
    [],                           // 28: tunnels (empty)
    [],                           // 29: tunnel limits (empty)
    [],                           // 30: clock times (empty)
    0,                            // 31: stronghold density
    10,                           // 32: stronghold minimum army
    25,                           // 33: stronghold maximum army
    [],                           // 34: stronghold indices (empty)
    [],                           // 35: stronghold armies (empty)
    [],                           // 36: clock timestamps (empty)
    null                          // 37: modifier options
];

writeFileSync("example.gior", compressToUint8Array(JSON.stringify(data)));
```

Opening `example.gior` in REPLAY provides four half-turns of playback with player names `RED` and `BLUE`. In the ALL view, the board progresses as follows; each number is the army on that cell:

```text
tick 0: [RED general: 1] [neutral: 0]   [neutral: 0] [BLUE general: 1]
tick 2: [RED general: 2] [neutral: 0]   [neutral: 0] [BLUE general: 2]
tick 3: [RED general: 1] [RED plain: 1] [neutral: 0] [BLUE general: 2]
tick 4: [RED general: 2] [RED plain: 1] [neutral: 0] [BLUE general: 3]
```

The two initial passes allow both generals to reach two soldiers at tick `2`. Red's full move sends one soldier to cell `1` and leaves one on its general. Blue's surrender at tick `3` ends the game after that half-turn's growth, producing the final position at tick `4` and the result `RED WINS`. Surrender preserves the remaining territory, so blue's general stays blue.

### Populated Example

The following code can be appended to the minimal example's JavaScript program. It copies that example's complete field layout, then supplies a larger map, custom initial armies, moves by both players, and chat and map-marker records. 

```javascript
const populated = structuredClone(data);
populated[1] = "populated";
populated[2] = 4;
populated[3] = 3;
populated[5] = [12, 8];
populated[6] = [2];
populated[7] = [4];
populated[8] = [0, 11];
populated[9] = [5, 6];
populated[10] = [
    [0, 0, 1, 0, 0],     // tick 0: red moves its full army right
    [1, 11, 10, 1, 0],   // tick 0: blue moves half its army left
    [0, 1, 2, 0, 1],     // tick 1: red captures the city
    [0, 2, 3, 1, 2]      // tick 2: red moves half its city army right
];
populated[11] = [[1, 3]];
populated[12] = [0, 1];
populated[13] = "example-map";
populated[14] = [0, 4, 11];
populated[15] = [12, 2, 8];
populated[17] = [["Ready.", "", 0, 0], ["Good game.", "", 1, 3]];
populated[26] = [[0, 1, 2], [1, 2, 10]];

writeFileSync("populated.gior", compressToUint8Array(JSON.stringify(populated)));
```

The initial board in the ALL view is

```text
[RED general: 12] [neutral: 0] [neutral city: 4] [neutral: 0]
[neutral: 2]      [mountain]   [mountain]        [neutral: 0]
[neutral: 0]      [neutral: 0] [neutral: 0]      [BLUE general: 8]
```

At tick `0`, red sends eleven soldiers to cell `1`, and blue sends four to cell `10`. At tick `1`, red sends ten soldiers from cell `1` into the four-soldier city, captures it with six remaining, and gains one city soldier from the growth that produces tick `2`. Red then sends three of its seven city soldiers to cell `3`. Blue surrenders at tick `3`; after the final growth, the replay ends at tick `4` with `RED WINS` and this board:

```text
[RED general: 3] [RED plain: 1] [RED city: 5]   [RED plain: 3]
[neutral: 2]     [mountain]      [mountain]     [neutral: 0]
[neutral: 0]     [neutral: 0]    [BLUE plain: 4] [BLUE general: 6]
```
