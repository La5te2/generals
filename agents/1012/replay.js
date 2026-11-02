// Adapter around the vendored official replay engine. State snapshots precede
// both players' recorded moves; recorded ordering is preserved during replay.
'use strict';
const readline = require('readline');
const Game = require('./vendor/replay/Game');

function replay(data, limit) {
    if (![5, 6, 7, 13].includes(data.version)) throw Error(`Unsupported replay version ${data.version}`);
    if (data.generals.length !== 2 || data.generals.some(x => x < 0)) throw Error('Expected two live generals');
    if (data.teams && data.teams.length && (data.teams.length !== 2 || data.teams[0] === data.teams[1])) throw Error('Team game');
    if (data.teams && data.teams.length === 0) data.teams = null;
    const area = data.mapWidth * data.mapHeight;
    const cells = [...data.generals, ...data.cities, ...data.mountains];
    if (cells.some(c => !Number.isInteger(c) || c < 0 || c >= area) || new Set(cells).size !== cells.length) {
        throw Error('Invalid or overlapping terrain');
    }
    if (data.cities.length !== data.cityArmies.length || data.cityArmies.some(a => !Number.isInteger(a) || a < 0 || a > 100000)) {
        throw Error('Invalid city armies');
    }
    for (const key of ['swamps', 'neutrals', 'modifiers', 'lights', 'observatories', 'lookouts']) {
        if (data[key] && data[key].length) throw Error(`Unsupported map field ${key}`);
    }
    // v13 standard 1v1 has the same combat, growth and recorded move semantics
    // as v6/7. Later chat/color metadata do not affect simulation. No current
    // move-priority rules are substituted for the recorded historical order.
    const moves = data.moves.map(m => Array.isArray(m) ?
        {index:m[0], start:m[1], end:m[2], is50:m[3], turn:m[4]} : m);
    const afks = (data.afks || []).map(a => Array.isArray(a) ? {index:a[0], turn:a[1]} : a);
    if ([...moves, ...afks].some(e => !Number.isInteger(e.turn) || e.turn < 0)) throw Error('Invalid event time');
    for (let i = 1; i < afks.length; ++i) {
        if (afks[i].turn < afks[i-1].turn) throw Error('AFKs are not chronological');
    }
    for (let i = 1; i < moves.length; ++i) {
        if (moves[i].turn < moves[i-1].turn) throw Error('Moves are not chronological');
    }
    const last = Math.max(0, ...moves.map(m => m.turn), ...afks.map(a => a.turn));
    if (last >= limit) throw Error(`Replay exceeds ${limit} half-turns`);
    const game = Game.createFromReplay(data);
    let at = 0, afkAt = 0, invalid = 0;
    const frames = [];
    while (!game.isOver() && game.turn <= last) {
        const actions = [-1, -1]; // -1 is pass, -2 is an unusable label.
        const frame = {tick:game.turn, owners:game.map._map.slice(), armies:game.map._armies.slice(),
                       generals:game.generals.slice(), cities:game.cities.slice(), actions};
        const counts = [0, 0];
        while (at < moves.length && moves[at].turn === game.turn) {
            const m = moves[at++];
            if (![0,1].includes(m.index) || ![0,1,false,true].includes(m.is50)) throw Error('Invalid move fields');
            if (!game.map.isAdjacent(m.start, m.end) || !game.map.isValidTile(m.start) || !game.map.isValidTile(m.end)) {
                invalid++; actions[m.index] = -2; continue;
            }
            const dr = Math.floor(m.end / data.mapWidth) - Math.floor(m.start / data.mapWidth);
            const dc = m.end % data.mapWidth - m.start % data.mapWidth;
            const direction = dr < 0 ? 0 : dr > 0 ? 1 : dc < 0 ? 2 : 3;
            actions[m.index] = counts[m.index]++ ? -2 : m.start * 8 + direction * 2 + Number(m.is50);
            if (game.handleAttack(m.index, m.start, m.end, m.is50) === false) {
                invalid++; actions[m.index] = -2;
            }
        }
        while (afkAt < afks.length && afks[afkAt].turn === game.turn) {
            const p = afks[afkAt++].index;
            if (![0,1].includes(p)) throw Error('Invalid AFK player');
            if (game.deaths.includes(game.sockets[p])) game.tryNeutralizePlayer(p);
            else { game.deaths.push(game.sockets[p]); game.alivePlayers--; }
            actions[p] = -2; // A surrender is not a policy pass demonstration.
        }
        frames.push(frame);
        game.update();
    }
    const winner = game.isOver() ? game.sockets.findIndex(s => !game.deaths.includes(s)) : -1;
    return {frames, invalid, winner};
}

readline.createInterface({input:process.stdin}).on('line', line => {
    try { const request = JSON.parse(line); console.log(JSON.stringify(replay(request.replay, request.limit))); }
    catch (error) { console.log(JSON.stringify({error:String(error.message)})); }
});
