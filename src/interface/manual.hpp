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
        "win 0-10   window size, 10 for full screen\n"
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
RED PLAYER and BLUE PLAYER accept strategy programs and their arguments. A blank player field selects human control. At least one player uses an external program. The folder button selects an executable.

DIRECTORY selects where games are recorded as .gior files. An empty directory field leaves recording disabled. The top-right arrow starts the match.

ONLINE
SERVER selects BOT or MAIN. Use the account registered on that server. Main-server automated play requires authorization for that account.

PLAYER accepts a strategy program, or stays blank for human control. USERNAME is the public account name. USER ID is the private account credential.

PRIVATE ROOM takes a room ID such as cikp. An empty field selects ranked 1v1 matchmaking.

PROXY takes an anonymous HTTP CONNECT address such as http://127.0.0.1:10090. Use the HTTP or mixed port configured in your proxy software. An empty field selects a direct connection.

The top-right arrow connects with the selected settings. Online timing and game results come from the server.

HUMAN CONTROLS
Click an owned tile to select it. Click a neighboring tile or press WASD or an arrow key to queue a move. Further moves will extend the queued route shown on the map.

Shift or a second click on the selected tile toggles half-army movement for the next move. G selects the general. Right-click clears the selection.

E removes the last queued move. Q clears the queued route. Space toggles trigger pause in local play.

PLAYBACK AND NAVIGATION
The pause button and Space toggle local play or replay playback. Step buttons operate while paused. Local program-versus-program play supports forward steps. Replay supports both directions, also using Left and Right.

Human local play offers pause. Online play follows the server continuously. Dimmed controls indicate actions unavailable in the current mode.

In local program-versus-program play and replay, keys 1, 2 and 3 select red, blue and full-board perspectives. Human play uses the controlled player's perspective.

Stop ends the match, connection or playback while keeping the last position visible. Reset becomes available after Stop or completion. It starts a fresh local game, reconnects online, or restarts the selected replay.

Stopping an ongoing local match with a human player surrenders the human player. With two programs, Stop compares army totals, then land totals, with blue winning an exact tie. The losing player surrenders, and the final half-turn completes its scheduled growth.

The back arrow or Escape returns one scene, first to configuration and then to the home page. Leaving a local match saves its recording when DIRECTORY is set.

REPLAY
REPLAY FILE selects a .gior recording from local play or a downloaded mainstream 1v1 game. The folder button opens a file picker, and the top-right arrow starts playback. The console's turn command adjusts playback speed.

CONSOLE
F1 toggles the console. Escape closes it while open. The game keeps its current running or paused state while reading this manual.

The mouse wheel, Up and Down scroll the output. Page Up and Page Down scroll by a page. Typing a new command keeps the manual visible until submission.

Commands are case-insensitive. Enter submits a command. Use help for the command list. turn uses a positive duration in milliseconds. Local play and replay start with 500 ms per half-turn.

auto 0 is the default and keeps Reset manual. auto 1 waits 1000 ms after completion or Stop, then resets the current mode. back and quit cancel a pending automatic reset. Errors await a manual retry.
)";
}
