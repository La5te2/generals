# Protocols

This document defines the external agent protocol, interface console commands, supported online messages, LAN room messages, and the GIOR replay format. Each section specifies the data exchanged, its meaning, and the order of communication.

## External Agents

### Program Commands

The Player field contains an executable followed by its arguments. Whitespace separates arguments; single or double quotes group an argument containing spaces. Quotes must be balanced, and backslashes remain literal. The command is passed directly to process creation: shell operators, variable expansion and redirection are not interpreted.

```text
"D:\Games\simple.exe" "D:\Models\policy.pt"
python -B "D:\Agents\agent.py" "D:\Models\policy.eqx"
```

An empty Player field selects human input instead of an external program. The recording directory is a path, not an executable command; its console syntax is defined under `RD` below.

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

`tick` counts half-turns. LOCAL and LAN start at `0`. Main- and bot-server sessions use the server's `turn` value, starting from `1`.

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

For main- and bot-server sessions, a move is eligible for submission while its observation tick matches the latest received board. Pass waits for the next observation, and replies to older ticks expire. Submitted moves remain pending until server confirmation. The server checks move legality at execution time and advances the processed index for both executed and discarded moves.

LAN agent replies also expire when their observation tick is no longer current. The host checks the tick again when it receives the move. Acknowledgment confirms receipt and processing of the request, not execution of the move; the following observations describe its outcome. A slow agent may skip intermediate observations. Pass submits no queued move.

Malformed replies, oversized lines, and premature process exit end the agent's session. In an ongoing LAN game, an agent failure also surrenders that player's side.

Each action line allows up to 255 bytes before LF. With CRLF endings, CR counts toward that limit. Fields are whitespace-separated integers, and each line follows the field count specified above.

LOCAL deadlines follow the configured half-turn duration. The LAN host advances the game at a fixed 500 ms interval, while main- and bot-server timing follows server updates. An agent sends its reply as soon as its decision is ready.

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

An external C++ agent can implement the same exchange through the standard process protocol.

## Interface Console

Each submission contains one command. Command names are case-insensitive, so `TURN 250` and `turn 250` are equivalent. Numeric arguments are whitespace-separated decimal integers. Paths retain their case, and `RD` requires the quoting described below. Extra arguments invalidate a command.

- `help` takes zero arguments and displays the command list.
- `man` takes zero arguments and displays the game and interface manual.
- `back` takes zero arguments and returns to the previous scene. From a match, it ends the session and returns to configuration.
- `quit` takes zero arguments and ends the session before closing the application. A recording save failure defers exit until the recording can be saved.
- `turn MS` accepts a positive integer and sets the LOCAL and REPLAY half-turn duration in milliseconds, regardless of the currently selected mode. It does not affect current or future ONLINE sessions: LAN uses a fixed 500 ms interval, while main- and bot-server timing follows the server.
- `win WIDTH HEIGHT` sets the window's content size. Width must be from `720` through `7680`, and height from `560` through `4320`. The result is limited to the monitor's usable area. `win 0 0` selects borderless fullscreen; a single zero is invalid.
- `rd "folder path"` selects the recording directory for new LOCAL and LAN games. `rd ""` disables saving for new games.
- `auto N` accepts `0` for manual Reset or `1` for automatic Reset after a 1000 ms delay following normal completion or manual Stop.

`auto` defaults to `0`. With automatic Reset enabled, LOCAL starts a new game, main- and bot-server sessions reconnect and join the queue, LAN starts fresh matchmaking after its room closes and recording cleanup finishes, and REPLAY restarts the current file. REPLAY has no Stop operation. `back` and `quit` cancel a pending automatic Reset. Sessions ending in an error await manual retry.

The recording directory is empty at application startup. Its value is shared across modes and is not persisted across application launches. The double quotes around the `RD` path are mandatory, even without spaces. Backslashes are literal path characters, not escape sequences; embedded double quotes are unsupported. A nonempty directory is created if necessary. An invalid directory leaves the previous setting unchanged. An ongoing game keeps the directory selected when it started. A completed recording whose write failed is retried immediately at the new directory, without starting another game. An empty directory cannot discard a pending recording. The path is local to each participant and is never sent to a LAN peer.

```text
win 1280 800
win 0 0
rd "D:\Games\Replays"
rd ""
```

## Online Servers

### Connection and Identity

Online sessions use the Engine.IO 4 WebSocket transport and the Socket.IO default namespace. The current endpoints are:

```text
Main: wss://ws.generals.io/socket.io/?EIO=4&transport=websocket
Bot:  wss://botws.generals.io/socket.io/?EIO=4&transport=websocket
```

The User ID identifies the account in join requests. The server supplies public usernames in `game_start`. A private-game request includes the room ID, while ranked play uses `join_1v1`.

The optional Proxy field accepts an anonymous HTTP CONNECT proxy, for example `http://127.0.0.1:7890`. The `http://` prefix and explicit port are required. Proxy credentials, paths and query strings are unsupported. An empty field uses a direct connection. A configured proxy does not fall back to a direct connection on failure. These proxy semantics also apply to LAN connections.

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

## LAN Rooms

### Network Discovery

Each window advertises on IPv4 multicast group `239.255.42.17`, UDP port `42817`, with TTL `1` and multicast loopback enabled. The IP configuration field selects a local adapter address without a port, not a remote peer. Empty selects all available IPv4 adapters. A physical or virtual LAN must carry this multicast traffic and allow connections between participants. There is no directory server, forwarding endpoint or NAT traversal.

The discovery datagram is a JSON object:

```json
{"type":"generals-lan","id":"0123456789abcdef0123456789abcdef","room":"","port":49152,"available":true,"host":""}
```

`id` is a fresh random 32-character hexadecimal instance ID for this matchmaking attempt. `port` is its independently allocated TCP listening port. The source IP of the datagram supplies the peer address. `available` means the instance is accepting pairing requests; an in-flight connection reserves it. Once paired, `host` identifies the accepting instance. Announcements repeat every 500 ms and when availability changes; unseen peers expire after three seconds. IDs distinguish windows, including several windows on the same computer.

An empty `room` participates in public matching. A nonempty ID identifies one private room within the discovery network and matches only the same case-sensitive ID, with 1-64 printable ASCII characters. An occupied private room reports `Room is full`; public participants skip occupied peers and wait or form another pair. Room IDs do not authenticate participants. Discovery is unauthenticated and belongs on a trusted network.

The optional HTTP CONNECT proxy applies to outgoing WebSocket connections, without direct fallback. It must be able to reach the discovered address. Discovery itself stays on the selected network and exposes that local adapter address; the proxy does not anonymize discovery.

### Pairing

Available instance IDs are ordered lexicographically and adjacent instances pair. The lower ID initiates the connection; the higher ID accepts and becomes the host. Private matching selects the first pair; an advertised in-flight reservation blocks another pair from claiming the same Room ID. Each window reserves at most one connection. Pairing uses these messages in order:

```json
{"type":"hello","id":"00000000000000000000000000000001","target":"00000000000000000000000000000002","room":"","name":"Bob"}
{"type":"offer","id":"00000000000000000000000000000002","target":"00000000000000000000000000000001","room":"","names":["Alice","Bob"]}
{"type":"accept","id":"00000000000000000000000000000001","target":"00000000000000000000000000000002"}
{"type":"paired"}
```

`name` must contain 1 through 64 printable ASCII characters. The receiver validates both instance IDs and the room. A reserved or unavailable participant replies with `{"type":"busy"}` and closes the extra connection. Pairing attempts time out after five seconds and release the reservation. Proxy failures report an error rather than retrying directly.

After confirmation, both listeners stop accepting new connections. The host owns seat `0` (red); the initiating participant owns seat `1` (blue). The host advances the engine once every 500 ms, independently of either participant's `turn` setting, and `names` remains in red/blue order. Each connection carries exactly one game. Other pairs do not depend on either participant.

Game messages are one JSON object per WebSocket text frame, at most 256 KiB each. Every message has a case-sensitive `type`. Binary frames and Engine.IO/Socket.IO prefixes are unsupported.

### Observations and Queues

During and after a game, the host sends `state` messages. The following example uses a complete three-by-three observation:

```json
{"type":"state","rows":3,"cols":3,"view":"8 2 6 3 12\n4 1 2\n1 1 1\n0 0 5\n1 1 0\n0 0 0\n0 0 0\n5 1 0\n0 0 0\n0 0 0\n","result":0,"queued":[],"ack":0}
```

`rows` and `cols` are the board dimensions. `view` is the complete external-agent observation text, without initialization; ownership is relative to the recipient. `result` is `0` for ongoing, `1` for red victory, `2` for blue victory, or `3` for a draw. Terminal states include the final observation. The guest receives only its own field of view, not the full board.

`queued` contains the receiving player's pending actions as five-integer strings. `ack` is the highest processed input ID, initially zero. It confirms processing of a queue edit, not execution of a move. States may repeat a tick when only the queue changes. Ticks cannot decrease; a game begins at tick zero.

### Actions

Queue edits carry a strictly increasing positive unsigned 64-bit `id` within this connection.

```json
{"type":"move","id":1,"action":"0 1 1 0 0\n"}
{"type":"move","id":2,"tick":8,"action":"0 1 1 3 1\n"}
{"type":"cancel","id":3,"all":false}
```

`action` uses the external-agent action format and may contain at most 100 bytes. Optional `tick` restricts acceptance to that exact host tick. Agent moves carry it; human queued moves omit it. Stale ticks are ignored but acknowledged. A later game uses a new connection, so old inputs cannot enter it.

`cancel` removes the newest pending move when `all` is false, or clears the queue when true. Move legality is checked against the game state, including at execution time. A correctly formatted but illegal move does not imply a transport failure.

### Completion and Room Lifetime

The following client commands have no additional required fields:

```json
{"type":"leave"}
{"type":"ping"}
```

Stop, Back and Quit request departure. `leave` surrenders an ongoing game and ends the room for both participants. Natural completion also ends the room. An orderly end transfers the final replay before closing; an abrupt disconnection may prevent delivery. A closed room never accepts a replacement player.

The interface may retain the final position. Reset or automatic Reset starts a new discovery and pairing attempt with the current configuration; it is not readiness within the previous room.

Both peers answer `ping` with `{"type":"pong"}`. Heartbeats are sent every five seconds; twenty seconds without incoming activity is a connection failure.

Orderly closure uses:

```json
{"type":"closed"}
```

The connection closes after these messages. Unexpected transport loss may close it without a final message.

### Replay Transfer

At completion, the host preserves the recording locally and sends the same compressed GIOR bytes to the guest in ordered chunks:

```json
{"type":"replay","offset":0,"data":"0012abff","last":false}
```

`data` is lowercase hexadecimal, with two characters per byte. `offset` counts decoded bytes, begins at zero and must equal the end of the previous chunk. A chunk contains at most 8192 bytes, and a complete transfer at most 16 MiB. `last` marks the final chunk. The example illustrates chunk framing, not a complete replay.

The recipient acknowledges completion with:

```json
{"type":"recorded"}
```

This acknowledgment confirms receipt, not successful disk persistence. Transfer and acknowledgment are required even when local recording is disabled. Each participant independently applies its `RD` setting. An orderly departure allows the completed replay to be delivered before closure, but transport loss can prevent delivery.

## Replays

### File Structure

LOCAL recordings, LAN recordings and downloaded games share the `.gior` format used by generals.io. The file contains a positional JSON array compressed with LZ-String's `compressToUint8Array`. Compressed 16-bit words appear as two bytes each, high byte first. Text decoding follows UTF-16 semantics, preserving Unicode player names.

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

In LAN, Stop or departure surrenders the requesting or disconnected player's side, whether human or external agent. Both recipients receive the same completed recording. No additional replay event is needed for a manual stop.

### Filenames

LOCAL and LAN recordings are saved to the directory selected through `RD`. Filenames consist of an eight-character lowercase hexadecimal digest followed by `.gior`. Downloaded files keep their original names. The digest is 32-bit FNV-1a over the compressed file bytes $b_0, \ldots, b_{n-1}$:

$$
h_0 = 2166136261,
\qquad
h_{k+1} = \bigl((h_k \mathbin{\oplus} b_k) \times 16777619\bigr) \bmod 2^{32}.
$$

Here, $\oplus$ denotes bitwise XOR. The final value $h_n$ becomes eight lowercase hexadecimal characters, padded with leading zeroes. The filename identifies the local recording, while the binary contents retain the GIOR structure.

An existing file with identical contents is reused. If the same digest names different contents, saving reports an error instead of overwriting that recording. The short digest is not a guarantee of uniqueness or authenticity.
