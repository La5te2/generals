// console reference text: keep the command list and game manual together for display inside the application.
#pragma once

#include <string_view>

namespace NEBULA::Manual {
    inline constexpr std::string_view commands =
        "help       command list\n"
        "man        game and interface manual\n"
        "back       previous scene\n"
        "quit       close the application\n"
        "turn MS    local and replay half-turn duration\n"
        "win W H    W: 720-7680, H: 560-4320; 0 0 full screen\n"
        "rd \"PATH\"  recording directory; \"\" disables\n"
        "auto 0/1   manual or automatic reset";

    // game introduction follows the official tutorial at https://generals.io/. controls below describe this interface.
    inline constexpr std::string_view text = 
R"(Generals is a real-time strategy game played on a grid. Build your army, expand your territory and find the enemy general while defending your own.

YOUR GENERAL
The crown marks your general. It recruits one soldier each turn. In a two-player game, capture the opposing general to win.

LAND AND ARMIES
Tile colors identify their owners. Each number shows the soldiers on that tile. Every 25 turns, each tile you own gains one soldier.

MOVEMENT AND COMBAT
Move to a neighboring tile horizontally or vertically, leaving one soldier behind. Half moves send half the army, rounded down. Friendly armies combine. When attacking, soldiers cancel each other out. An attack stronger than the defenders captures the tile with its survivors.

CITIES AND MOUNTAINS
Capture a city by overcoming its defending army. An owned city recruits one soldier per turn, adding to your supply. Mountains block movement.

FOG OF WAR
Your territory reveals nearby tiles. Expand to discover the map and locate your opponent. In fog, cities and mountains share an obstacle symbol. The scoreboard shows both players' total army and land.

LOCAL
RED PLAYER and BLUE PLAYER are command fields: enter a program path followed by its arguments. Quote each path or argument containing spaces, for example "D:/My Agents/nebula.exe" "D:/My Models/best.pt". A blank player field selects human control. At least one player uses an external program. The folder button fills in the executable path, and arguments can be appended afterward.

The top-right arrow starts the match. Recording uses the console's global RD setting, described below.

ONLINE
SERVER selects BOT, MAIN or LAN. For BOT and MAIN, use the account registered on that server. Main-server automated play requires authorization for that account.

PLAYER uses the same command format as LOCAL, including program arguments and quoted paths, or stays blank for human control. USERNAME is the public account name. USER ID is the private account credential.

PRIVATE ROOM takes a room ID such as cikp. An empty field selects ranked 1v1 matchmaking.

PROXY takes an anonymous HTTP CONNECT address such as http://127.0.0.1:10090. Use the HTTP or mixed port configured in your proxy software. An empty field selects a direct connection.

The top-right arrow connects with the selected settings. Online timing and game results come from the server.

LAN
USERNAME is your display name. USER ID becomes ROOM ID, and PRIVATE ROOM becomes IP. PLAYER remains a program command or blank for human control. LAN does not use your BOT or MAIN credentials.

PROXY uses the same http://host:port format as BOT and MAIN. Outgoing game connections use this route without direct fallback. The peer's LAN address must be reachable from the proxy. Discovery packets still use the selected local network and expose the local adapter address; PROXY does not make LAN discovery anonymous.

IP selects this computer's local IPv4 adapter, for example 192.168.1.10, without a port. Use the adapter address of the desired physical or virtual LAN. Empty selects all available adapters. Both players must share a network that carries IPv4 multicast discovery and permits direct WebSocket connections. Each window gets its own listening port automatically. There is no central server, forwarding address or NAT traversal. Use a trusted network; Room IDs are not passwords or authentication.

A blank Room ID matches another available public player. Other public players keep waiting or form separate pairs. A nonempty ID matches only the same case-sensitive ID; an occupied private room reports Room is full. A paired room cannot accept another player.

The connection roles are selected automatically. The accepting player is marked [HOST], plays red, and runs the match using their turn setting. The other player plays blue. Each player sees only their own observation. Stop surrenders and closes the room for both players. Back and Quit also leave the room. Natural completion closes it after the replay transfer. The final position remains on screen; Reset starts fresh matchmaking with the current configuration.

Each pair runs independently. Closing one window does not affect any other pair.

HUMAN CONTROLS
Click an owned tile to select it. Click a neighboring tile or press WASD or an arrow key to queue a move. Further moves will extend the queued route shown on the map.

Shift or a second click on the selected tile toggles half-army movement for the next move. G selects the general. Right-click clears the selection.

E removes the last queued move. Q clears the queued route. Space toggles trigger pause in local play.

PLAYBACK AND NAVIGATION
The pause button and Space toggle local play or replay playback. Step buttons operate while paused. Local program-versus-program play supports forward steps. Replay supports both directions; holding Left or Right repeats steps. After reaching the end, Left can still step backward.

Human local play offers pause. Online play follows the server continuously. Dimmed controls indicate actions unavailable in the current mode.

In local program-versus-program play and replay, keys 1, 2 and 3 select red, blue and full-board perspectives. Human play uses the controlled player's perspective.

Stop ends a local match, LAN room or BOT/MAIN connection while keeping the last position visible. It stays disabled in replay. Reset becomes available after completion and recording cleanup. It starts a fresh local game, starts matchmaking again, or restarts the selected replay.

Stopping an ongoing local match with a human player surrenders the human player. With two programs, Stop compares army totals, then land totals, with blue winning an exact tie. The losing player surrenders, and the final half-turn completes its scheduled growth.

The back arrow or Escape returns one scene, first to configuration and then to the home page. Leaving a local or LAN match saves its recording when RD is set.

RECORDING
RD "folder path" sets a global recording directory for new LOCAL and LAN matches, for example RD "D:/My Replays". The directory starts empty each time the program launches. RD "" disables recording for new matches. Changes do not alter the recording destination of an already-started game. Recordings are saved as eight-character digest filenames with a .gior extension.

LOCAL saves at completion or interruption. LAN transfers the complete replay after completion, Stop or an orderly departure; each player saves it to their own RD directory. An unexpected loss of the host or network can prevent delivery. During play the guest never receives a full-map recording. If saving fails, RD with a valid directory retries the pending recording without starting another match.

REPLAY
REPLAY FILE selects a .gior recording from local play or a downloaded mainstream 1v1 game. The folder button opens a file picker, and the top-right arrow starts playback. The console's turn command adjusts playback speed.

CONSOLE
F1 toggles the console. Escape closes it while open. The game keeps its current running or paused state while reading this manual.

The mouse wheel, Up and Down scroll the output. Page Up and Page Down scroll by a page. Typing a new command keeps the manual visible until submission.

Commands are case-insensitive. Enter submits a command. Use help for the command list. turn uses a positive duration in milliseconds and affects only LOCAL and REPLAY, which start with 500 ms per half-turn. ONLINE is unaffected: LAN uses a fixed 500 ms interval, and BOT/MAIN follow their server's timing.

win W H sets the window's content width and height. Width accepts 720-7680 and height accepts 560-4320; dimensions are capped by the current monitor's usable area. win 0 0 selects borderless full screen. For example, win 1280 900 returns to a window of that size when the monitor has enough room. A single zero is not valid.

auto 0 is the default and keeps Reset manual. auto 1 waits 1000 ms after completion and recording cleanup, then resets the current mode. In LAN this starts matchmaking for a new room. back and quit cancel a pending automatic reset. Errors await a manual retry.
)";
}
