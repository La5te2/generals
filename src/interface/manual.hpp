// Console reference strings: help lists commands; man explains game rules, configuration and controls.
#pragma once

#include <string_view>

namespace NEBULA::Manual {
    inline constexpr std::string_view commands =
        "help       command list\n"
        "man        game and interface manual\n"
        "back       return to the previous page\n"
        "quit       close the application\n"
        "turn MS    LOCAL / REPLAY milliseconds per half-turn\n"
        "win W H    W: 720-7680, H: 560-4320. 0 0 full screen\n"
        "rd \"PATH\"  LOCAL / LAN recording folder. \"\" disables\n"
        "auto 0/1   automatic restart off / on";

    inline constexpr std::string_view text =
R"(Generals is a real-time strategy game played on a grid. Expand your territory to build an army, find the opposing general and capture it while protecting your own.

THE BOARD
Red and blue tiles belong to their respective players. A number on a visible tile is its army. The crown marks a general, the city symbol marks a city, and mountains block movement. The scoreboard shows each player's total army and number of owned tiles, including territory hidden by fog.

TIME AND REINFORCEMENTS
A turn consists of two half-turns. Each player can make one move per half-turn. Your general and each city you own produce one soldier per turn. Every 25 turns, every tile you own gains one additional soldier.

MOVES AND CAPTURES
Move from a tile you own with at least two soldiers to a horizontally or vertically adjacent tile. A full move sends all but one soldier. A half move sends half the source army, rounded down. Soldiers sent to another tile you own join its army.

On a neutral or enemy tile, attacking and defending soldiers cancel each other out. If the attacking army is larger, you capture the tile with the surviving attackers. Otherwise, the defender keeps the tile with its remaining soldiers. Equal armies leave it with zero. Capture a neutral city by overcoming its garrison.

Capturing the opposing general normally wins the game. Its tile becomes a city, and the defeated player's remaining territory transfers to the winner with those armies halved, rounded up.

If both players successfully capture the opposing general in the same half-turn, they exchange generals and their other territories. Armies on the other tiles are halved, rounded up, and play continues.

FOG OF WAR
Each tile you own reveals itself and its eight surrounding neighbors, including diagonals. Outside that view, fog hides ownership and armies. Hidden cities and mountains share an obstacle symbol, so an unexplored obstacle may be a city you can capture or a mountain you cannot cross.

STARTING A GAME
Choose LOCAL, ONLINE or REPLAY on the home page, then fill in that mode's configuration. START MATCH begins a local game, CONNECT starts online connection or LAN matchmaking, and PLAY opens a replay.

PLAYER FIELDS
A player field selects who controls that player. Leave it blank for keyboard and mouse control, or enter an external agent's launch command. An agent can use any implementation language, provided its program exchanges observations and actions through the project's agent protocol.

A launch command consists of a program path or name followed by the arguments that program requires. Matching single or double quotes group a path or argument containing spaces. Backslashes are literal. The command is launched directly, without interpreting shell operators, variable expansion or redirection.

The folder button selects a program file and fills the player field with its quoted path. Add any required arguments after that path. Model paths and other arguments depend on the selected agent.

LOCAL
RED PLAYER controls red. BLUE PLAYER controls blue. Each uses the player-field format above. At least one field must select an external agent, because this window provides keyboard and mouse control for one player.

START MATCH creates a new random map. With a human player, the board shows that player's view. With two agents, RED, BLUE and ALL select either player's view or the full board. The default local interval is 500 ms per half-turn. TURN changes this interval.

ONLINE
SERVER selects BOT, MAIN or LAN. BOT connects to the bot server. MAIN connects to the main server. LAN matches players on a shared local or virtual network.

PLAYER selects your external agent or human control. The remaining configuration fields depend on the selected server, as described below.

BOT AND MAIN
USERNAME is the public name of your account on the selected server. USER ID is that account's private credential and should be kept confidential. The name and credential must identify the same account on that server.

PRIVATE ROOM selects a server room by its ID, for example cikp. Enter the room ID itself. Leave this field empty for ranked 1v1 matchmaking.

CONNECT joins the selected room or matchmaking queue. The server assigns your color, controls the game clock and reports the result. During play and after completion, this window shows the latest view received for your player.

LAN
USERNAME is the display name shown to your opponent. It accepts 1-64 printable ASCII characters.

ROOM ID selects a private room. Both players enter the same case-sensitive ID, up to 64 printable ASCII characters. An occupied private room reports Room is full. Leave ROOM ID empty for public matchmaking: available players form pairs, while unmatched players keep waiting. Once paired, a room accepts no additional players. Room IDs identify matches but provide no authentication, so use a trusted network.

IP selects a network adapter on your own computer. Enter that adapter's IPv4 address, for example 192.168.1.10, without a port. To use a particular physical or virtual LAN, enter your address on that network. Leave IP empty to search through all usable local adapters.

Both players need a shared network that permits IPv4 multicast discovery and WebSocket connections between them. Each window searches for a matching opponent and establishes a peer connection automatically. Multiple windows can form independent pairs, including on the same computer.

The window marked [HOST] runs the match and plays red. The other player plays blue. These roles are assigned automatically. Each player sees their own view during the match. LAN uses a fixed 500 ms half-turn interval.

When the match ends or either player leaves, the room closes for both players. A completed recording provides the final RED, BLUE and ALL views. The board remains on screen. Reset searches for a new opponent with the current configuration.

PROXY
PROXY selects an HTTP CONNECT proxy for outgoing BOT, MAIN or LAN game connections. Enter http://host:port, for example http://127.0.0.1:10090, using the HTTP or mixed port provided by your proxy software. The address must specify a port and omit authentication credentials. Leave it empty for a direct connection.

With a proxy configured, outgoing game connections use that proxy without falling back to a direct connection. For LAN, the proxy must be able to reach the opponent's LAN address. LAN discovery still runs on the selected local network and exposes your local adapter address to that network. The proxy does not anonymize discovery.

REPLAY
REPLAY FILE is the path to a .gior recording. Enter the path itself, including any spaces, without enclosing quotes, or select it with the folder button. PLAY opens the recording. Recordings from LOCAL and LAN and supported two-player Generals server replays use this format.

RED, BLUE and ALL select a player's recorded view or the full board. The replay starts playing automatically. Pause it to step through individual half-turns. TURN sets the playback interval.

HUMAN CONTROLS
Click a tile you own to select it. Click an adjacent tile or press W, A, S, D or an arrow key to queue a move. The selection advances to the destination, allowing further moves to extend the route shown on the board. The game executes queued moves as half-turns advance.

Press Shift or click the selected tile again to toggle half-army movement for the next queued move. The 50% indicator shows when it is enabled. After queuing that move, movement returns to full army. G selects your general. Right-click clears the selection while preserving the queued route.

E removes the last queued move and selects its source. Q clears all queued moves. Space pauses or resumes LOCAL. Moves can still be queued while it is paused.

PAUSE, STEP AND VIEW
The play/pause button or Space pauses and resumes LOCAL or REPLAY. BOT, MAIN and LAN continue according to their game clocks. A dimmed button is unavailable in the current state.

While paused, a LOCAL match between two agents can advance one half-turn with the forward-step button or the period key. A paused REPLAY can step backward or forward with its step buttons, comma and period, or Left and Right. Holding Left or Right repeats replay steps. After the replay ends, backward steps remain available.

When RED, BLUE and ALL are available, keys 1, 2 and 3 select those views. They are available during agent-versus-agent LOCAL play, during REPLAY, after a LOCAL match, and after a LAN match with a complete recording. An ongoing human match stays on the controlled player's view.

STOP AND RESET
Stop ends the current LOCAL match, leaves a LAN room or closes a BOT/MAIN session. In LAN, leaving an ongoing match surrenders your side and closes the room for both players. Stop is unavailable in REPLAY.

In an ongoing LOCAL match with a human player, Stop surrenders the human player. With two agents, the player with less total army surrenders. If armies are equal, the player with less land surrenders. If both are equal, red surrenders. The final half-turn completes its scheduled growth.

Completion and Stop leave the last position on screen. Reset becomes available when the session finishes. It creates a new LOCAL game, reconnects or starts matching again in ONLINE, or restarts the selected REPLAY. A recording awaiting a successful save must be saved before a new LOCAL or LAN match can start.

BACK AND QUIT
The back arrow, Escape with the console closed, or BACK returns from the board to its configuration page, then from configuration to the home page. Leaving an ongoing match ends your participation. In LOCAL it applies the Stop result. In LAN it closes the room for both players.

QUIT or closing the window exits the application. LOCAL and LAN recordings are saved when a recording folder was set for that match. If saving fails, the application stays open so that the recording can be saved to another folder.

RECORDING
RD "folder path" selects the recording folder for new LOCAL and LAN matches. For example, RD "D:/My Replays" records subsequent matches there. The setting starts empty when the application launches. RD "" disables recording for new matches.

Each match retains the recording folder selected when it began. Changing RD during play affects later matches. Recordings are saved as .gior files with eight-character digest names.

LOCAL saves its recording when the match ends, including Stop or leaving the board. LAN delivers the complete recording to both players after completion or an orderly departure. Each saves to their own folder. An unexpected loss of the host or network can prevent the other player from receiving the recording.

If a save fails, enter RD with a valid folder to retry the pending recording without starting another match. For LAN, wait until the room has finished closing before retrying.

CONSOLE
F1 opens or closes the console. Escape closes an open console. Opening it leaves the game's running or paused state unchanged.

Type a command and press Enter. Command names are case-insensitive. HELP lists commands. MAN displays this manual. The mouse wheel, Up and Down scroll the output. Page Up and Page Down scroll by a page. Submitting another command replaces the manual with that command's output.

TURN MS sets a positive number of milliseconds per half-turn for LOCAL and REPLAY. The initial setting is 500. It changes local game speed or replay playback speed and remains in effect for subsequent LOCAL and REPLAY sessions. BOT and MAIN use server timing. LAN remains at 500 ms.

WIN W H sets the window's content width and height. Width accepts 720-7680. Height accepts 560-4320. The resulting window is limited to the monitor's usable area. WIN 0 0 selects borderless full screen. Use two positive dimensions to return to windowed mode, for example WIN 1280 900. Both dimensions must be zero for full screen.

AUTO 0 disables automatic Reset and is the default. AUTO 1 enables it: after a session finishes successfully, the application waits 1000 ms and starts a new LOCAL game, reconnects or matches again in ONLINE, or restarts the REPLAY. BACK and QUIT cancel a pending automatic restart. An error requires a manual retry.
)";
}
