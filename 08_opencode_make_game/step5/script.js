const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const hpDisplay = document.getElementById('hp');
const enemyCountDisplay = document.getElementById('enemy-count');
const seedDisplay = document.getElementById('seed-val');
const messageDisplay = document.getElementById('message');
const overlay = document.getElementById('overlay');
const resultText = document.getElementById('result-text');

const TILE_SIZE = 40;
const GRID_SIZE = 10;

const ENEMY_TYPES = [
    { name: 'Slime', symbol: 's', color: '#4f4', hp: 5, atk: 1, weight: 0.7 },
    { name: 'Orc', symbol: 'O', color: '#f44', hp: 12, atk: 3, weight: 0.3 },
];

const ITEM_TYPES = [
    { name: 'Healing Potion', symbol: '!', color: '#f0f', weight: 0.6, effect: (p) => {
        const heal = 8;
        p.hp = Math.min(p.maxHp, p.hp + heal);
        return `You drank a Healing Potion! HP restored. (HP: ${p.hp}/${p.maxHp})`;
    }},
    { name: 'Strength Potion', symbol: '+', color: '#ff0', weight: 0.25, effect: (p) => {
        p.atk += 1;
        return `You drank a Strength Potion! ATK increased to ${p.atk}.`;
    }},
    { name: 'Health Fruit', symbol: '*', color: '#0f0', weight: 0.15, effect: (p) => {
        p.maxHp += 5;
        p.hp += 5;
        return `You ate a Health Fruit! Max HP increased to ${p.maxHp}. (HP: ${p.hp}/${p.maxHp})`;
    }},
];

let MAP = [];
let player = {};
let enemies = [];
let items = [];
let messageHistory = [];

// Seeded Random
let randomFunc = Math.random;
function setupRandom() {
    const params = new URLSearchParams(window.location.search);
    const seedStr = params.get('seed');
    if (seedStr) {
        const seed = parseInt(seedStr, 10);
        seedDisplay.innerText = seed;
        randomFunc = mulberry32(seed);
    } else {
        const seed = Math.floor(Math.random() * 1000000);
        seedDisplay.innerText = seed;
        randomFunc = mulberry32(seed);
    }
}

function mulberry32(a) {
    return function() {
        let t = a += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

function initGame() {
    setupRandom();

    let reachableTiles;
    do {
        generateMap();
        spawnPlayer();
        reachableTiles = ensureReachability();
    } while (reachableTiles.length < 10); // Ensure a minimum playable area

    spawnEnemies();
    spawnItems();
    draw();
}

function generateMap() {
    MAP = [];
    for (let y = 0; y < GRID_SIZE; y++) {
        MAP[y] = [];
        for (let x = 0; x < GRID_SIZE; x++) {
            if (x === 0 || x === GRID_SIZE - 1 || y === 0 || y === GRID_SIZE - 1) {
                MAP[y][x] = 1;
            } else {
                MAP[y][x] = randomFunc() < 0.2 ? 1 : 0;
            }
        }
    }
}

function spawnPlayer() {
    let placed = false;
    while (!placed) {
        const x = Math.floor(randomFunc() * (GRID_SIZE - 2)) + 1;
        const y = Math.floor(randomFunc() * (GRID_SIZE - 2)) + 1;
        if (MAP[y][x] === 0) {
            player = {
                x: x,
                y: y,
                hp: 20,
                maxHp: 20,
                atk: 3,
                symbol: '@',
                color: '#44f'
            };
            placed = true;
        }
    }
}

function ensureReachability() {
    const reachable = [];
    const queue = [{x: player.x, y: player.y}];
    const visited = new Set([`${player.x},${player.y}`]);

    while (queue.length > 0) {
        const {x, y} = queue.shift();
        reachable.push({x, y});
        for (const [dx, dy] of [[0,1],[0,-1],[1,0],[-1,0]]) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && MAP[ny][nx] === 0 && !visited.has(`${nx},${ny}`)) {
                visited.add(`${nx},${ny}`);
                queue.push({x: nx, y: ny});
            }
        }
    }

    for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
            if (MAP[y][x] === 0 && !visited.has(`${x},${y}`)) {
                MAP[y][x] = 1;
            }
        }
    }
    return reachable;
}

function spawnEnemies() {
    enemies = [];
    const reachableTiles = [];
    const visited = new Set();
    const queue = [{x: player.x, y: player.y}];
    visited.add(`${player.x},${player.y}`);
    while(queue.length > 0) {
        const {x, y} = queue.shift();
        reachableTiles.push({x, y});
        for (const [dx, dy] of [[0,1],[0,-1],[1,0],[-1,0]]) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && MAP[ny][nx] === 0 && !visited.has(`${nx},${ny}`)) {
                visited.add(`${nx},${ny}`);
                queue.push({x: nx, y: ny});
            }
        }
    }

    const enemyCount = 3 + Math.floor(randomFunc() * 3);
    let spawned = 0;
    let attempts = 0;

    while (spawned < enemyCount && attempts < 500) {
        attempts++;
        const tile = reachableTiles[Math.floor(randomFunc() * reachableTiles.length)];
        const x = tile.x;
        const y = tile.y;

        const dist = Math.abs(x - player.x) + Math.abs(y - player.y);
        if (dist >= 4 && !enemies.some(e => e.x === x && e.y === y)) {
            const type = getRandomEnemyType();
            enemies.push({
                ...type,
                x: x,
                y: y,
                hp: type.hp,
                maxHp: type.hp
            });
            spawned++;
        }
    }
}

function spawnItems() {
    items = [];
    const reachableTiles = [];
    const visited = new Set();
    const queue = [{x: player.x, y: player.y}];
    visited.add(`${player.x},${player.y}`);
    
    const tilesByDist = [];

    while(queue.length > 0) {
        const {x, y} = queue.shift();
        const dist = Math.abs(x - player.x) + Math.abs(y - player.y);
        reachableTiles.push({x, y, dist});
        if (dist <= 10) {
            tilesByDist.push({x, y});
        }
        for (const [dx, dy] of [[0,1],[0,-1],[1,0],[-1,0]]) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && MAP[ny][nx] === 0 && !visited.has(`${nx},${ny}`)) {
                visited.add(`${nx},${ny}`);
                queue.push({x: nx, y: ny});
            }
        }
    }

    const itemCount = 3 + Math.floor(randomFunc() * 3);
    let spawned = 0;

    // Guaranteed item within distance 10
    const nearTiles = tilesByDist.filter(t => t.x !== player.x || t.y !== player.y);
    if (nearTiles.length > 0) {
        const tile = nearTiles[Math.floor(randomFunc() * nearTiles.length)];
        if (!enemies.some(e => e.x === tile.x && e.y === tile.y)) {
            const type = getRandomItemType();
            items.push({ ...type, x: tile.x, y: tile.y });
            spawned++;
        }
    }

    // Remaining items
    let attempts = 0;
    while (spawned < itemCount && attempts < 500) {
        attempts++;
        const tile = reachableTiles[Math.floor(randomFunc() * reachableTiles.length)];
        const x = tile.x;
        const y = tile.y;

        if ((x !== player.x || y !== player.y) && 
            !enemies.some(e => e.x === x && e.y === y) && 
            !items.some(i => i.x === x && i.y === y)) {
            const type = getRandomItemType();
            items.push({ ...type, x: x, y: y });
            spawned++;
        }
    }
}

function getRandomItemType() {
    const rand = randomFunc();
    let cumulativeWeight = 0;
    for (const type of ITEM_TYPES) {
        cumulativeWeight += type.weight;
        if (rand < cumulativeWeight) return type;
    }
    return ITEM_TYPES[0];
}

function getRandomEnemyType() {
    const rand = randomFunc();
    let cumulativeWeight = 0;
    for (const type of ENEMY_TYPES) {
        cumulativeWeight += type.weight;
        if (rand < cumulativeWeight) return type;
    }
    return ENEMY_TYPES[0];
}

function addMessage(msg) {
    messageHistory.push(msg);
    if (messageHistory.length > 3) messageHistory.shift();
    messageDisplay.innerText = messageHistory.join('\n');
}

function getPath(start, end) {
    const queue = [[start]];
    const visited = new Set([`${start.x},${start.y}`]);
    while (queue.length > 0) {
        const path = queue.shift();
        const {x, y} = path[path.length - 1];
        if (x === end.x && y === end.y) return path;
        for (const [dx, dy] of [[0,1],[0,-1],[1,0],[-1,0]]) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && MAP[ny][nx] === 0 && !visited.has(`${nx},${ny}`)) {
                visited.add(`${nx},${ny}`);
                queue.push([...path, {x: nx, y: ny}]);
            }
        }
    }
    return null;
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
            if (MAP[y][x] === 1) {
                ctx.fillStyle = '#777';
                ctx.fillRect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            } else {
                ctx.strokeStyle = '#333';
                ctx.strokeRect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE);
            }
        }
    }
    enemies.forEach(enemy => {
        ctx.fillStyle = enemy.color;
        ctx.fillRect(enemy.x * TILE_SIZE + 4, enemy.y * TILE_SIZE + 4, TILE_SIZE - 8, TILE_SIZE - 8);
        ctx.fillStyle = '#000';
        ctx.font = '20px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(enemy.symbol, enemy.x * TILE_SIZE + TILE_SIZE / 2, enemy.y * TILE_SIZE + TILE_SIZE / 2);
    });
    items.forEach(item => {
        ctx.fillStyle = item.color;
        ctx.font = '20px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(item.symbol, item.x * TILE_SIZE + TILE_SIZE / 2, item.y * TILE_SIZE + TILE_SIZE / 2);
    });
    ctx.fillStyle = player.color;
    ctx.fillRect(player.x * TILE_SIZE + 4, player.y * TILE_SIZE + 4, TILE_SIZE - 8, TILE_SIZE - 8);
    ctx.fillStyle = '#000';
    ctx.font = '20px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(player.symbol, player.x * TILE_SIZE + TILE_SIZE / 2, player.y * TILE_SIZE + TILE_SIZE / 2);
    hpDisplay.innerText = player.hp;
    enemyCountDisplay.innerText = enemies.length;
}

function checkItemPickup() {
    const itemIndex = items.findIndex(i => i.x === player.x && i.y === player.y);
    if (itemIndex !== -1) {
        const item = items[itemIndex];
        const msg = item.effect(player);
        addMessage(msg);
        items.splice(itemIndex, 1);
    }
}

function movePlayer(dx, dy) {
    const nx = player.x + dx;
    const ny = player.y + dy;
    if (nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE) return;
    if (MAP[ny][nx] === 1) return;
    const enemy = enemies.find(e => e.x === nx && e.y === ny);
    if (enemy) {
        attackEnemy(enemy);
    } else {
        player.x = nx;
        player.y = ny;
        checkItemPickup();
    }
    enemyTurn();
    draw();
    checkGameOver();
}

function attackEnemy(enemy) {
    enemy.hp -= player.atk;
    addMessage(`Attack! ${enemy.name} takes ${player.atk} damage.`);
    if (enemy.hp <= 0) {
        addMessage(`${enemy.name} defeated!`);
        const index = enemies.indexOf(enemy);
        enemies.splice(index, 1);
    }
}

function enemyTurn() {
    for (const enemy of enemies) {
        if (player.hp <= 0) break;

        const path = getPath({x: enemy.x, y: enemy.y}, {x: player.x, y: player.y});
        if (!path || path.length - 1 > 8) continue;

        const nextStep = path[1];
        if (nextStep.x === player.x && nextStep.y === player.y) {
            player.hp -= enemy.atk;
            addMessage(`${enemy.name} attacks! You take ${enemy.atk} damage.`);
        } else {
            const occupied = enemies.some(e => e !== enemy && e.x === nextStep.x && e.y === nextStep.y);
            if (!occupied) {
                enemy.x = nextStep.x;
                enemy.y = nextStep.y;
            }
        }
    }
}

function checkGameOver() {
    if (player.hp <= 0) {
        showResult('GAME OVER');
    } else if (enemies.length === 0) {
        showResult('VICTORY!');
    }
}

function showResult(text) {
    resultText.innerText = text;
    overlay.classList.remove('hidden');
}

window.addEventListener('keydown', e => {
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) {
        e.preventDefault();
    }
    if (overlay.classList.contains('hidden')) {
        switch (e.key) {
            case 'ArrowUp': movePlayer(0, -1); break;
            case 'ArrowDown': movePlayer(0, 1); break;
            case 'ArrowLeft': movePlayer(-1, 0); break;
            case 'ArrowRight': movePlayer(1, 0); break;
        }
    }
});

initGame();
