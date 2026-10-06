# Protocols

This document defines the external agent protocol, interface console commands, supported online messages, and the GIOR replay format. Each section specifies the data exchanged, its meaning, and the order of communication.

## External Agents

### Streams and Lifecycle

An agent receives initialization and observations through `stdin`, sends actions through `stdout`, and writes diagnostics to `stderr`.

An action is one line of integers followed by a newline. The agent flushes `stdout` immediately after each reply so that the session can read it before the half-turn deadline. Agent commands are `Move` and `Pass`. Interface commands such as `help` and `back` belong to the GUI console.

Each connection serves one game. Initialization appears once, followed by observations and replies. EOF on `stdin` signals the end of the game.

### Initialization

The first line contains three decimal integers:

```text
player rows cols
```

`player` is the agent's seat, with `0` for red and `1` for blue. Let $H$ denote `rows` and $W$ denote `cols`. Both dimensions remain fixed throughout the game and satisfy

$$
1 \le H \le 40, \qquad 1 \le W \le 40.
$$

For example, `1 20 23` assigns the agent to blue on a board with 20 rows and 23 columns.

### Observations

Each observation contains one statistics line followed by three matrices, for a total of $1 + 3H$ lines. Each matrix has $H$ rows of exactly $W$ integers, ordered from top to bottom and left to right.

```text
tick my_land my_army opp_land opp_army
<H rows of terrain codes>
<H rows of ownership codes>
<H rows of army counts>
```

`tick` counts half-turns. LOCAL starts at `0`, while ONLINE uses the server's `turn` value, starting from `1`.

`my_land` and `my_army` give the agent's total land and army. `opp_land` and `opp_army` give the opponent's public totals. These statistics cover each player's entire territory, while the matrices describe the agent's current field of view.

Terrain codes are:

- `0`: fog.
- `1`: visible plain.
- `2`: visible mountain.
- `3`: visible city.
- `4`: visible general.
- `5`: a hidden obstacle, representing a mountain or city under fog.

Ownership is relative to the receiving agent: `0` means neutral or hidden, `1` means self, and `2` means opponent. Thus, both red and blue programs identify their own cells using `owner == 1` on the wire.

Army counts use signed 64-bit integers in the range $0 \le A \le 2^{63}-1$. Cells with terrain code `0` or `5` carry zero ownership and army values as placeholders for hidden information. On visible cells, an army value of `0` represents an actual empty garrison.

The following observation describes a three-by-three board. The agent's general is at `(0, 0)` with five soldiers, and its other cell has one soldier. The opponent's public totals are three tiles and twelve soldiers.

```text
8 2 6 3 12
4 1 2
1 1 1
0 0 5
1 1 0
0 0 0
0 0 0
5 1 0
0 0 0
0 0 0
```

### Actions

The agent replies to each observation with exactly five decimal integers:

```text
kind row col direction split
```

`kind` selects `0` for Move or `1` for Pass. For a move, `row` and `col` identify the source cell. Coordinates start at `(0, 0)` in the top-left corner, with rows increasing downward and columns increasing rightward.

`direction` selects `0` for up, `1` for down, `2` for left, or `3` for right. `split` selects `0` for a full move or `1` for a half move. Given source army $A \ge 2$ at execution time and split flag $s$, the number sent is

$$
S(A, s) =
\begin{cases}
A - 1 & s = 0, \\
\left\lfloor A/2 \right\rfloor & s = 1.
\end{cases}
$$

For example, a source with nine soldiers sends eight on a full move or four on a half move. The standard Pass line is `1 0 0 0 0`. The final four integers are placeholders for Pass.

```text
0 4 7 0 0
0 4 7 3 1
1 0 0 0 0
```

These are three independent replies: a full move upward from `(4, 7)`, a half move rightward from the same cell, and a pass.

Move legality is determined by the board at execution time. A legal move starts from an owned cell with at least two soldiers and targets an adjacent cell within the board that the rules allow it to enter.

### Timing and Failures

The exchange follows a request/reply sequence. The agent reads one complete observation, replies with exactly one action, and then reads the next observation. Each reply belongs to the tick of the preceding observation.

For LOCAL, a matching reply received by the deadline participates in that half-turn. If the deadline arrives while the session is still waiting, the action for that half-turn becomes Pass. A slow agent may receive observations with gaps in their tick values.

For ONLINE, a move is eligible for submission while its observation tick matches the latest received board. Pass waits for the next observation, and replies to older ticks expire. Submitted moves remain pending until server confirmation. The server checks move legality at execution time and advances the processed index for both executed and discarded moves. Malformed replies, oversized lines, and premature process exit end the session.

Each action line allows up to 255 bytes before LF. With CRLF endings, CR counts toward that limit. Fields are whitespace-separated integers, and each line follows the field count specified above.

LOCAL deadlines follow the configured half-turn duration. ONLINE action timing follows server updates. An agent sends its reply as soon as its decision is ready.

### Minimal Agent

This Python program consumes complete observations and replies with Pass. An agent can replace the final output with a calculated action while retaining the same input sequence and immediate flush.

```python
import sys

def read_grid(rows):
    return [list(map(int, sys.stdin.readline().split())) for row in range(rows)]

initial = sys.stdin.readline()
if initial:
    player, rows, cols = map(int, initial.split())
    for header in sys.stdin:
        tick, my_land, my_army, opp_land, opp_army = map(int, header.split())
        terrain = read_grid(rows)
        owner = read_grid(rows)
        army = read_grid(rows)
        print("1 0 0 0 0", flush=True)
```

[simple.cpp](../src/agents/simple.cpp) provides a complete C++ example of the same exchange.

## Interface Console

Each submission contains one command with whitespace-separated arguments. Command names are case-insensitive, so `TURN 250` and `turn 250` are equivalent.

- `help` takes zero arguments and displays the command list.
- `man` takes zero arguments and displays the game and interface manual.
- `back` takes zero arguments and returns to the previous scene. From a match, it ends the session and returns to configuration.
- `quit` takes zero arguments and ends the session before closing the application. A recording save failure defers exit until the recording can be saved.
- `turn MS` accepts a positive integer and sets the LOCAL and REPLAY half-turn duration in milliseconds.
- `win N` accepts an integer from `0` through `10` and selects a fixed window size. Level `10` fills the current screen with a borderless window.
- `auto N` accepts `0` for manual Reset or `1` for automatic Reset after a 1000 ms delay following normal completion or manual Stop.

`auto` defaults to `0`. With automatic Reset enabled, LOCAL starts a new game, ONLINE reconnects and joins the queue, and REPLAY restarts the current file. `back` and `quit` cancel a pending automatic Reset. Sessions ending in an error await manual retry.

## Online Servers

### Connection and Identity

Online sessions use the Engine.IO 4 WebSocket transport and the Socket.IO default namespace. The current endpoints are:

```text
Main: wss://ws.generals.io/socket.io/?EIO=4&transport=websocket
Bot:  wss://botws.generals.io/socket.io/?EIO=4&transport=websocket
```

The User ID identifies the account in join requests. The server supplies public usernames in `game_start`. A private-game request includes the room ID, while ranked play uses `join_1v1`.

### Message Framing

Numeric prefixes describe the transport packet, and the string inside an event array names the game event. Connection setup proceeds as follows:

1. The server sends `0{...}` with the Engine.IO session identifier and heartbeat parameters.
2. The client sends `40` to connect to the Socket.IO default namespace.
3. The server confirms with `40{"sid":"..."}`.
4. During the connection, the client answers each Engine.IO ping `2` with pong `3`.

Game events use `42["event", ...]`. An acknowledgment identifier appears between the prefix and the payload. For example, `420["queue_count"]` requests queue information with acknowledgment ID `0`, and `430[...]` carries its response. Packet `41` disconnects the default namespace.

### Joining and Moving

The following events cover queue queries, account lookup, and game entry. The examples use symbolic `USER_ID`, `ROOM`, and `CLIENT_KEY` values. `CLIENT_KEY` is a public client identifier, while `USER_ID` is the account credential.

Bot-server account lookup and ranked entry use:

```text
42["stars_and_rank", USER_ID, null]
42["join_1v1", USER_ID, null, 0]
```

Main-server account lookup and ranked entry use:

```text
42["stars_and_rank", USER_ID]
42["join_1v1", USER_ID, CLIENT_KEY, 0, null, true]
```

Private-game entry uses the corresponding `join_private` event:

```text
Bot:  42["join_private", ROOM, USER_ID]
Main: 42["join_private", ROOM, USER_ID, CLIENT_KEY, null]
```

Movement uses `42["attack", from, to, half, index]`. The `half` argument is a JSON boolean. The client assigns increasing positive integer indices within each game, including after cancellation. Source and destination use row-major indices, where $r$ is the row, $c$ is the column, and $W$ is the board width:

$$
i = rW + c, \qquad 0 \le r < H, \quad 0 \le c < W.
$$

`42["cancel"]` leaves the matchmaking queue, and `42["leave_game"]` leaves a game.

`42["undo_move"]` removes the newest queued move on the server. `42["clear_moves"]` clears the server's movement queue. These events affect moves still awaiting execution when the server receives them.

### Game Events

`queue_update` reports queue status, and `pre_game_start` announces an upcoming game. `game_start` supplies `playerIndex`, the `usernames` array, and game options. This section covers mainstream 1v1 with two opposing players, using indices `0` for red and `1` for blue.

Within `game_update`, `turn` is the half-turn counter. `map_diff` patches the previous map array, and `cities_diff` patches the previous city-index array. `generals` gives both general indices, with `-1` for a hidden general. Entries in `scores` identify players through `i` and provide land through `tiles` and army through `total`.

`attackIndex` identifies the latest processed move. Pending moves with indices at or below this value have been processed, including moves discarded as illegal. Subsequent board data describes the actual outcome. A missing or null `attackIndex` leaves the previous confirmation in effect.

A diff alternates a count of old values to copy with a count of replacement values and the replacements themselves. A trailing copy count can stand alone. Both copied and replaced runs advance the cursor in the old array. For example:

```text
previous: [10, 20, 30, 40]
diff:     [1, 2, 21, 31, 1]
result:   [10, 21, 31, 40]
```

The decoded map contains `width`, `height`, an army layer, and an ownership/terrain layer in that order. Each layer contains $HW$ row-major values. Supported mainstream maps may also carry an appended all-zero tunnel layer. This layer participates in subsequent map diffs. In the ownership/terrain layer, `0` and `1` identify the players, `-1` marks neutral land, `-2` marks mountains, `-3` marks fog, and `-4` marks hidden obstacles. City and general indices refine the visible terrain.

Board updates describe the account's perspective, while scores provide both players' public totals. Diffs apply in arrival order, including updates with a repeated `turn`.

`game_won` and `game_lost` describe victory and defeat relative to the account.

## Replays

### File Structure

Local recordings and downloaded games share the `.gior` format used by generals.io. The file contains a positional JSON array compressed with LZ-String's `compressToUint8Array`. Compressed 16-bit words appear as two bytes each, high byte first. Text decoding follows UTF-16 semantics, preserving Unicode player names.

The current writer emits format revision `19`. The reader accepts revisions `15` through `19` for mainstream 1v1. These revisions share the movement priorities used by the local engine. Supported games contain two opposing players, plains, mountains, cities and generals.

The initial array fields are:

```text
0   format revision
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
```

An index addresses a cell in row-major order. City indices and armies have matching lengths. Generals start with one soldier, and the army override arrays replace the initial army on their listed cells.

Later fields carry special terrain, chat, colors, settings, modifiers, transforms, pings, general trades and clock metadata. The field layout follows the [generals.io replay serializer](https://generals.io/generals-main-prod-v31.4.3-485dbf55.js). Mainstream recordings use empty arrays for special terrain and modifiers. Format revision `19` occupies 38 array positions.

### Moves

Each replay move has five integer fields:

```text
[player, from, to, half, tick]
```

`player` selects `0` for red or `1` for blue. `from` and `to` identify adjacent cells, and `half` selects `0` for a full move or `1` for a half move. `tick` identifies the position before the move executes, beginning at `0`. Records appear in increasing tick order, with at most one move per player per half-turn. A half-turn with an empty move list represents both players passing.

The agent protocol also contains five integers, with a different purpose and field order:

```text
kind row column direction split
```

The replay reader translates between cell indices and the engine's coordinates and direction. Player identity and tick come from the recording session. Existing agent replies retain their original format.

### Endings

General captures and time limits derive their results from the game rules. A surrender event has the form `[player, tick]`. It ends that player's participation before the moves at `tick`, and the half-turn completes its scheduled growth. In 1v1, the remaining player wins.

Stopping an ongoing local human match records the human player's surrender. With two programs, Stop compares army totals, then land totals, with blue winning an exact tie. The losing player surrenders at the next unresolved half-turn. The resulting file uses the existing surrender event.

### Filenames

Local filenames consist of an eight-character lowercase hexadecimal digest followed by `.gior`. Downloaded files keep their original names. The local digest is 32-bit FNV-1a over the compressed file bytes $b_0, \ldots, b_{n-1}$:

$$
h_0 = 2166136261,
\qquad
h_{k+1} = \bigl((h_k \mathbin{\oplus} b_k) \times 16777619\bigr) \bmod 2^{32}.
$$

Here, $\oplus$ denotes bitwise XOR. The final value $h_n$ becomes eight lowercase hexadecimal characters, padded with leading zeroes. The filename identifies the local recording, while the binary contents retain the GIOR structure.
