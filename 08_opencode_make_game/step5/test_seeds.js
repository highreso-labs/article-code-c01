const GRID_SIZE = 10;
const ENEMY_TYPES = [
    { name: 'Slime', symbol: 's', color: '#4f4', hp: 5, atk: 1, weight: 0.7 },
    { name: 'Orc', symbol: 'O', color: '#f44', hp: 12, atk: 3, weight: 0.3 },
];
const ITEM_TYPES = [
    { name: 'Healing Potion', symbol: '!', color: '#f0f', weight: 0.6 },
    { name: 'Strength Potion', symbol: '+', color: '#ff0', weight: 0.25 },
    { name: 'Health Fruit', symbol: '*', color: '#0f0', weight: 0.15 },
];

let MAP = [];
let player = {};
let enemies = [];
let items = [];
let randomFunc = Math.random;

function mulberry32(a) {
    return function() {
        let t = a += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
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

function keepLargestComponent() {
    const visited = new Set();
    let largestComponent = [];

    for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
            if (MAP[y][x] === 0 && !visited.has(`${x},${y}`)) {
                const component = [];
                const queue = [{x, y}];
                visited.add(`${x},${y}`);

                while (queue.length > 0) {
                    const {x, y} = queue.shift();
                    component.push({x, y});
                    for (const [dx, dy] of [[0,1],[0,-1],[1,0],[-1,0]]) {
                        const nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && MAP[ny][nx] === 0 && !visited.has(`${nx},${ny}`)) {
                            visited.add(`${nx},${ny}`);
                            queue.push({x: nx, y: ny});
                        }
                    }
                }
                if (component.length > largestComponent.length) {
                    largestComponent = component;
                }
            }
        }
    }

    const largestSet = new Set(largestComponent.map(t => `${t.x},${t.y}`));
    for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
            if (MAP[y][x] === 0 && !largestSet.has(`${x},${y}`)) {
                MAP[y][x] = 1;
            }
        }
    }
    return largestComponent;
}

function spawnPlayer(reachableTiles) {
    const tile = reachableTiles[Math.floor(randomFunc() * reachableTiles.length)];
    player = {
        x: tile.x,
        y: tile.y,
        hp: 20,
        maxHp: 20,
        atk: 3,
        symbol: '@',
        color: '#44f'
    };
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

function getRandomItemType() {
    const rand = randomFunc();
    let cumulativeWeight = 0;
    for (const type of ITEM_TYPES) {
        cumulativeWeight += type.weight;
        if (rand < cumulativeWeight) return type;
    }
    return ITEM_TYPES[0];
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

    const nearTiles = tilesByDist.filter(t => t.x !== player.x || t.y !== player.y);
    if (nearTiles.length > 0) {
        const tile = nearTiles[Math.floor(randomFunc() * nearTiles.length)];
        if (!enemies.some(e => e.x === tile.x && e.y === tile.y)) {
            const type = getRandomItemType();
            items.push({ ...type, x: tile.x, y: tile.y });
            spawned++;
        }
    }

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

function runGameInit(initialSeed) {
    let seed = initialSeed;
    let success = false;
    let finalSeed = seed;

    for (let attempt = 0; attempt < 20; attempt++) {
        randomFunc = mulberry32(seed);

        generateMap();
        const reachable = keepLargestComponent();
        if (reachable.length === 0) {
            seed++;
            continue;
        }

        spawnPlayer(reachable);
        spawnEnemies();
        spawnItems();

        if (reachable.length >= 20 && enemies.length > 0) {
            success = true;
            finalSeed = seed;
            break;
        }
        seed++;
    }
    
    return {
        success,
        finalSeed,
        reachableSize: (success) ? 
            (function() {
                let count = 0;
                for(let y=0; y<GRID_SIZE; y++) {
                    for(let x=0; x<GRID_SIZE; x++) {
                        if(MAP[y][x] === 0) count++;
                    }
                }
                return count;
            })() : 0,
        enemyCount: enemies.length
    };
}

// Test seed 794800
console.log('Testing seed 794800...');
const result794800 = runGameInit(794800);
console.log(result794800);

// Test seeds 1-100
console.log('\nTesting seeds 1-100...');
let failures = 0;
for (let i = 1; i <= 100; i++) {
    const res = runGameInit(i);
    if (!res.success) {
        console.log(`Seed ${i} failed!`);
        failures++;
    }
}
console.log(`Finished testing seeds 1-100. Failures: ${failures}`);
