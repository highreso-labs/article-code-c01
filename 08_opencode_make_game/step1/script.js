const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const hpDisplay = document.getElementById('hp');
const enemyCountDisplay = document.getElementById('enemy-count');
const messageDisplay = document.getElementById('message');
const overlay = document.getElementById('overlay');
const resultText = document.getElementById('result-text');

const TILE_SIZE = 40;
const GRID_SIZE = 10;

// Map definition: 1 = Wall, 0 = Floor
const MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

const player = {
    x: 1,
    y: 1,
    hp: 20,
    maxHp: 20,
    atk: 3,
    symbol: '@',
    color: '#44f'
};

const enemies = [
    { x: 8, y: 1, hp: 5, maxHp: 5, atk: 1, symbol: 's', color: '#4f4' },
    { x: 8, y: 8, hp: 5, maxHp: 5, atk: 1, symbol: 's', color: '#4f4' },
    { x: 1, y: 8, hp: 5, maxHp: 5, atk: 1, symbol: 's', color: '#4f4' },
    { x: 5, y: 5, hp: 5, maxHp: 5, atk: 1, symbol: 's', color: '#4f4' },
];

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

    // Draw enemies
    enemies.forEach(enemy => {
        ctx.fillStyle = enemy.color;
        ctx.fillRect(enemy.x * TILE_SIZE + 4, enemy.y * TILE_SIZE + 4, TILE_SIZE - 8, TILE_SIZE - 8);
        ctx.fillStyle = '#000';
        ctx.font = '20px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(enemy.symbol, enemy.x * TILE_SIZE + TILE_SIZE / 2, enemy.y * TILE_SIZE + TILE_SIZE / 2);
    });

    // Draw player
    ctx.fillStyle = player.color;
    ctx.fillRect(player.x * TILE_SIZE + 4, player.y * TILE_SIZE + 4, TILE_SIZE - 8, TILE_SIZE - 8);
    ctx.fillStyle = '#000';
    ctx.font = '20px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(player.symbol, player.x * TILE_SIZE + TILE_SIZE / 2, player.y * TILE_SIZE + TILE_SIZE / 2);

    // Update status
    hpDisplay.innerText = player.hp;
    enemyCountDisplay.innerText = enemies.length;
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
    }

    enemyTurn();
    draw();
    checkGameOver();
}

function attackEnemy(enemy) {
    enemy.hp -= player.atk;
    messageDisplay.innerText = `Attack! Slime takes ${player.atk} damage.`;
    if (enemy.hp <= 0) {
        messageDisplay.innerText = `Slime defeated!`;
        const index = enemies.indexOf(enemy);
        enemies.splice(index, 1);
    }
}

function enemyTurn() {
    enemies.forEach(enemy => {
        const dx = player.x - enemy.x;
        const dy = player.y - enemy.y;

        let moveX = 0;
        let moveY = 0;

        // Simple AI: Move closer to player
        if (Math.abs(dx) > Math.abs(dy)) {
            moveX = dx > 0 ? 1 : -1;
        } else if (Math.abs(dy) > 0) {
            moveY = dy > 0 ? 1 : -1;
        }

        const nx = enemy.x + moveX;
        const ny = enemy.y + moveY;

        if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE && MAP[ny][nx] === 0) {
            if (nx === player.x && ny === player.y) {
                // Attack player
                player.hp -= enemy.atk;
                messageDisplay.innerText = `Slime attacks! You take ${enemy.atk} damage.`;
            } else {
                // Check if another enemy is there
                const occupied = enemies.some(e => e !== enemy && e.x === nx && e.y === ny);
                if (!occupied) {
                    enemy.x = nx;
                    enemy.y = ny;
                }
            }
        }
    });
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
    if (overlay.classList.contains('hidden')) {
        switch (e.key) {
            case 'ArrowUp': movePlayer(0, -1); break;
            case 'ArrowDown': movePlayer(0, 1); break;
            case 'ArrowLeft': movePlayer(-1, 0); break;
            case 'ArrowRight': movePlayer(1, 0); break;
        }
    }
});

draw();
