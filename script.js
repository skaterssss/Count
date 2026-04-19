const categories = [
    {
        label: '1 tot 10',
        emoji: '🔢',
        hint: '1, 2, 3 … 10',
        color: 'linear-gradient(135deg, #f59e0b, #f97316)',
        shadow: '#c2570a',
        numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    },
    {
        label: '10 tot 20',
        emoji: '🔟',
        hint: '11, 12, 13 … 20',
        color: 'linear-gradient(135deg, #3b82f6, #6366f1)',
        shadow: '#1d4ed8',
        numbers: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    },
    {
        label: 'Even',
        emoji: '✌️',
        hint: '2, 4, 6 … 20',
        color: 'linear-gradient(135deg, #10b981, #059669)',
        shadow: '#047857',
        numbers: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
    {
        label: 'Oneven',
        emoji: '☝️',
        hint: '1, 3, 5 … 19',
        color: 'linear-gradient(135deg, #f43f5e, #e11d48)',
        shadow: '#be123c',
        numbers: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    },
    {
        label: 'Tientallen',
        emoji: '💯',
        hint: '10, 20, 30 … 100',
        color: 'linear-gradient(135deg, #8b5cf6, #a855f7)',
        shadow: '#6d28d9',
        numbers: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
];

// Images at 2× the max display size (500px container → 1000px) for retina screens.
// thumbUrl() generates a smaller variant for the collection grid cards.
const animals = [
    { name: 'Hond',    emoji: '🐶', image: 'https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Kat',     emoji: '🐱', image: 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Panda',   emoji: '🐼', image: 'https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Leeuw',   emoji: '🦁', image: 'https://images.unsplash.com/photo-1546182990-dffeafbe841d?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Olifant', emoji: '🐘', image: 'https://images.unsplash.com/photo-1564760055775-d63b17a55c44?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Giraf',   emoji: '🦒', image: 'https://images.unsplash.com/photo-1547721064-da6cfb341d50?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Pinguïn', emoji: '🐧', image: 'https://images.unsplash.com/photo-1551986782-d0169b3f8fa7?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Konijn',  emoji: '🐰', image: 'https://images.unsplash.com/photo-1585110396000-c9ffd4e4b308?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Vos',     emoji: '🦊', image: 'https://images.unsplash.com/photo-1474511320723-9a56873867b5?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Uil',     emoji: '🦉', image: 'https://images.unsplash.com/photo-1568393691622-c7ba131d63b4?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Zebra',   emoji: '🦓', image: 'https://images.unsplash.com/photo-1551969014-7d2c4cddf0b6?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Beer',    emoji: '🐻', image: 'https://images.unsplash.com/photo-1589656966895-2f33e7653819?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Eend',    emoji: '🦆', image: 'https://images.unsplash.com/photo-1518384401463-7c0f195d5ce1?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Koe',     emoji: '🐮', image: 'https://images.unsplash.com/photo-1516467508483-a7212febe31a?w=1000&h=1000&fit=crop&q=80' },
    { name: 'Dolfijn', emoji: '🐬', image: 'https://images.unsplash.com/photo-1607153333879-c174d265f1d2?w=1000&h=1000&fit=crop&q=80' },
];

function thumbUrl(url) {
    return url.replace('w=1000&h=1000&fit=crop&q=80', 'w=400&h=400&fit=crop&q=75');
}

// ── Collection (localStorage) ─────────────────────────────────────────────

function getCollection() {
    try {
        const saved = localStorage.getItem('dierenVerzameling');
        return saved ? new Set(JSON.parse(saved)) : new Set();
    } catch {
        return new Set();
    }
}

function addToCollection(animalName) {
    const col = getCollection();
    col.add(animalName);
    try { localStorage.setItem('dierenVerzameling', JSON.stringify([...col])); } catch {}
    updateTrophyButton();
}

function updateTrophyButton() {
    document.getElementById('collectedCount').textContent = getCollection().size;
}

// ── DOM references ─────────────────────────────────────────────────

const categoryScreen   = document.getElementById('categoryScreen');
const categoryGrid     = document.getElementById('categoryGrid');
const gameContent      = document.getElementById('gameContent');
const collectionScreen = document.getElementById('collectionScreen');
const numbersGrid      = document.getElementById('numbersGrid');
const message          = document.getElementById('message');
const resetBtn         = document.getElementById('resetBtn');
const backBtn          = document.getElementById('backBtn');
const trophyBtn        = document.getElementById('trophyBtn');
const collectionBackBtn= document.getElementById('collectionBackBtn');
const progressBar      = document.getElementById('progressBar');
const progressText     = document.getElementById('progressText');
const animalCover      = document.getElementById('animalCover');
const animalImage      = document.getElementById('animalImage');
const targetNumberEl   = document.getElementById('targetNumber');
const targetDisplay    = document.getElementById('targetDisplay');

// ── Game state ─────────────────────────────────────────────────────

let currentCategory = null;
let currentIndex    = 0;
let correctAnswers  = 0;
let currentAnimal   = null;

// ── Screen navigation ───────────────────────────────────────────────

function showScreen(name) {
    categoryScreen.style.display   = name === 'category'   ? 'flex' : 'none';
    gameContent.style.display      = name === 'game'       ? 'flex' : 'none';
    collectionScreen.style.display = name === 'collection' ? 'flex' : 'none';
}

// ── Category screen ────────────────────────────────────────────────

function buildCategoryScreen() {
    categories.forEach(cat => {
        const btn = document.createElement('button');
        btn.className = 'category-btn';
        btn.style.background = cat.color;
        btn.style.boxShadow = `0 5px 0 ${cat.shadow}`;
        btn.innerHTML =
            `<span class="cat-emoji">${cat.emoji}</span>` +
            `<span class="cat-info">` +
                `<span class="cat-label">${cat.label}</span>` +
                `<span class="cat-hint">${cat.hint}</span>` +
            `</span>`;
        btn.addEventListener('click', () => startGame(cat));
        categoryGrid.appendChild(btn);
    });
}

// ── Collection screen ──────────────────────────────────────────────

function showCollectionScreen() {
    showScreen('collection');
    buildCollectionGrid();
}

function buildCollectionGrid() {
    const col = getCollection();
    document.getElementById('collectionCount').textContent =
        `${col.size}/${animals.length}`;

    const grid = document.getElementById('collectionGrid');
    grid.innerHTML = '';

    animals.forEach(animal => {
        const card = document.createElement('div');
        card.className = col.has(animal.name) ? 'animal-card collected' : 'animal-card locked';

        if (col.has(animal.name)) {
            const img = document.createElement('img');
            img.src     = thumbUrl(animal.image);
            img.alt     = animal.name;
            img.loading = 'lazy';
            const nameEl = document.createElement('div');
            nameEl.className   = 'animal-card-name';
            nameEl.textContent = animal.name;
            card.appendChild(img);
            card.appendChild(nameEl);
        } else {
            card.textContent = '?';
        }

        grid.appendChild(card);
    });
}

// ── Game ─────────────────────────────────────────────────────────

function startGame(category) {
    currentCategory = category;
    showScreen('game');
    initGame();
}

function initGame() {
    currentIndex   = 0;
    correctAnswers = 0;
    currentAnimal  = animals[Math.floor(Math.random() * animals.length)];

    animalImage.innerHTML = '<div class="loading">Laden… ' + currentAnimal.emoji + '</div>';

    const img = new Image();
    img.onload = () => {
        animalImage.innerHTML = '';
        animalImage.appendChild(img);
    };
    img.onerror = () => {
        animalImage.innerHTML = '<div class="animal-emoji">' + currentAnimal.emoji + '</div>';
    };
    img.src = currentAnimal.image;
    img.alt = currentAnimal.name;

    // Snap cover back without transition to prevent flash of new image
    animalCover.style.transition = 'none';
    animalCover.style.clipPath   = 'inset(0 0 0 0)';
    document.querySelector('.animal-container').classList.remove('won');
    requestAnimationFrame(() => requestAnimationFrame(() => {
        animalCover.style.transition = '';
    }));

    updateProgress();
    targetNumberEl.textContent  = currentCategory.numbers[0];
    targetDisplay.style.display = 'flex';
    createNumberButtons();
    message.textContent = '';
    message.className   = 'message';
}

function createNumberButtons() {
    numbersGrid.innerHTML = '';
    const numbers = [...currentCategory.numbers];
    shuffleArray(numbers);

    numbers.forEach(num => {
        const btn = document.createElement('button');
        btn.className = 'number-btn';
        if (num >= 100) btn.classList.add('wide-number');
        btn.textContent      = num;
        btn.dataset.number   = num;
        btn.addEventListener('click', handleNumberClick);
        numbersGrid.appendChild(btn);
    });
}

function shuffleArray(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
}

function handleNumberClick(event) {
    const clicked  = parseInt(event.target.dataset.number);
    const expected = currentCategory.numbers[currentIndex];
    if (clicked === expected) {
        handleCorrectAnswer(event.target);
    } else {
        handleWrongAnswer(event.target);
    }
}

function handleCorrectAnswer(button) {
    correctAnswers++;
    currentIndex++;
    button.classList.add('correct');
    button.disabled = true;
    updateProgress();
    revealAnimal();

    if (correctAnswers === currentCategory.numbers.length) {
        addToCollection(currentAnimal.name);
        targetDisplay.style.display = 'none';
        message.textContent = '🎉 Geweldig! Je hebt gewonnen! 🌟';
        message.className   = 'message win';
        document.querySelector('.animal-container').classList.add('won');
        document.querySelectorAll('.number-btn').forEach(b => b.disabled = true);
    } else {
        targetNumberEl.textContent = currentCategory.numbers[currentIndex];
        message.textContent = '✅ Goed zo!';
        message.className   = 'message success';
    }
}

function handleWrongAnswer(button) {
    button.classList.add('wrong');
    setTimeout(() => button.classList.remove('wrong'), 500);
    message.textContent = '❌ Oeps! Probeer het opnieuw!';
    message.className   = 'message error';
}

function updateProgress() {
    const total = currentCategory.numbers.length;
    const pct   = (correctAnswers / total) * 100;
    progressBar.style.width    = pct + '%';
    progressText.textContent   = `${correctAnswers} / ${total}`;
}

function revealAnimal() {
    const pct = (correctAnswers / currentCategory.numbers.length) * 100;
    animalCover.style.clipPath = `inset(0 0 ${pct}% 0)`;
}

// ── Event listeners ────────────────────────────────────────────────

resetBtn.addEventListener('click', initGame);
backBtn.addEventListener('click', () => showScreen('category'));
trophyBtn.addEventListener('click', showCollectionScreen);
collectionBackBtn.addEventListener('click', () => showScreen('category'));

// ── Boot ──────────────────────────────────────────────────────────

buildCategoryScreen();
updateTrophyButton();
showScreen('category');
