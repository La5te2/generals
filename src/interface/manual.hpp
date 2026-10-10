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

PROXY uses the same http://host:port format as BOT and MAIN. It applies to outgoing WebSocket connections, not to the listening address. Empty means direct connection. With a proxy configured, a failed connection does not open a local lobby or retry directly. The target IP must be reachable from the proxy.

Enter IP:port, for example 192.168.1.10:8080. The program connects to that lobby, or opens one if the address is local and available. Empty IP uses a lobby on this machine at port 8080, opening a listener on all local interfaces if necessary. It does not discover other machines. Remote players enter the lobby machine's LAN IP, or a forwarded WebSocket address such as ws://example.net:18080. A wss:// forwarding endpoint is also accepted; the local listener uses unencrypted WebSocket. Room IDs are not passwords and do not authenticate players. Use a trusted network or a secured tunnel.

A blank Room ID joins an available public room or creates one and waits for another player. Public rooms receive internal numbers. A nonempty ID creates or joins a private room with that exact, case-sensitive ID. A full private room rejects further joins; public matching instead uses another room. Each room has two seats, and multiple independent rooms share the lobby IP.

The first player in each room is its host, marked [HOST], and plays red. The guest plays blue. The host's turn setting at room creation determines the half-turn duration. The lobby's local engine settles the match and sends each player only their own observation. Both players must be ready to start another match; Reset marks you ready. Stop surrenders the match but keeps you in the room. Leaving as guest frees the seat; leaving as host closes that room and notifies the guest. Back and Quit leave the room. Finishing a match alone does not close it.

The lobby is independent of any one room. Its process keeps serving other rooms after its own player returns to configuration or Home. Closing that process disconnects every room using its IP.

HUMAN CONTROLS
Click an owned tile to select it. Click a neighboring tile or press WASD or an arrow key to queue a move. Further moves will extend the queued route shown on the map.

Shift or a second click on the selected tile toggles half-army movement for the next move. G selects the general. Right-click clears the selection.

E removes the last queued move. Q clears the queued route. Space toggles trigger pause in local play.

PLAYBACK AND NAVIGATION
The pause button and Space toggle local play or replay playback. Step buttons operate while paused. Local program-versus-program play supports forward steps. Replay supports both directions; holding Left or Right repeats steps. After reaching the end, Left can still step backward.

Human local play offers pause. Online play follows the server continuously. Dimmed controls indicate actions unavailable in the current mode.

In local program-versus-program play and replay, keys 1, 2 and 3 select red, blue and full-board perspectives. Human play uses the controlled player's perspective.

Stop ends a local match or a BOT/MAIN connection while keeping the last position visible. It stays disabled in replay. Reset becomes available after Stop or completion. It starts a fresh local game, reconnects online, or restarts the selected replay. LAN keeps its room until its host leaves, as described above.

Stopping an ongoing local match with a human player surrenders the human player. With two programs, Stop compares army totals, then land totals, with blue winning an exact tie. The losing player surrenders, and the final half-turn completes its scheduled growth.

The back arrow or Escape returns one scene, first to configuration and then to the home page. Leaving a local or LAN match saves its recording when RD is set.

RECORDING
RD "folder path" sets a global recording directory for new LOCAL and LAN matches, for example RD "D:/My Replays". The directory starts empty each time the program launches. RD "" disables recording for new matches. Changes do not alter the recording destination of an already-started game. Recordings are saved as eight-character digest filenames with a .gior extension.

LOCAL saves at completion or interruption. LAN transfers the complete replay to both players after completion, Stop or an orderly departure; each player saves it to their own RD directory. An unexpected loss of the lobby or network can prevent delivery of the final replay. During play, clients never receive its full-map recording.

REPLAY
REPLAY FILE selects a .gior recording from local play or a downloaded mainstream 1v1 game. The folder button opens a file picker, and the top-right arrow starts playback. The console's turn command adjusts playback speed.

CONSOLE
F1 toggles the console. Escape closes it while open. The game keeps its current running or paused state while reading this manual.

The mouse wheel, Up and Down scroll the output. Page Up and Page Down scroll by a page. Typing a new command keeps the manual visible until submission.

Commands are case-insensitive. Enter submits a command. Use help for the command list. turn uses a positive duration in milliseconds. Local play and replay start with 500 ms per half-turn.

win W H sets the window's content width and height. Width accepts 720-7680 and height accepts 560-4320; dimensions are capped by the current monitor's usable area. win 0 0 selects borderless full screen. For example, win 1280 900 returns to a window of that size when the monitor has enough room. A single zero is not valid.

auto 0 is the default and keeps Reset manual. auto 1 waits 1000 ms after completion or Stop, then resets the current mode. back and quit cancel a pending automatic reset. Errors await a manual retry.
)";
}
