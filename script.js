// Game state
let currentNumber = 1;
let correctAnswers = 0;
const totalNumbers = 10;

// Array of animals with free images (Unsplash & Pixabay - vrij te gebruiken in NL)
const animals = [
    { name: 'Hond', emoji: '🐶', image: 'https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=600&h=600&fit=crop' },
    { name: 'Kat', emoji: '🐱', image: 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=600&h=600&fit=crop' },
    { name: 'Panda', emoji: '🐼', image: 'https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=600&h=600&fit=crop' },
    { name: 'Leeuw', emoji: '🦁', image: 'https://images.unsplash.com/photo-1546182990-dffeafbe841d?w=600&h=600&fit=crop' },
    { name: 'Olifant', emoji: '🐘', image: 'https://images.unsplash.com/photo-1564760055775-d63b17a55c44?w=600&h=600&fit=crop' },
    { name: 'Giraf', emoji: '🦒', image: 'https://images.unsplash.com/photo-1547721064-da6cfb341d50?w=600&h=600&fit=crop' },
    { name: 'Pinguïn', emoji: '🐧', image: 'https://images.unsplash.com/photo-1551986782-d0169b3f8fa7?w=600&h=600&fit=crop' },
    { name: 'Konijn', emoji: '🐰', image: 'https://images.unsplash.com/photo-1585110396000-c9ffd4e4b308?w=600&h=600&fit=crop' },
    { name: 'Vos', emoji: '🦊', image: 'https://images.unsplash.com/photo-1474511320723-9a56873867b5?w=600&h=600&fit=crop' },
    { name: 'Uil', emoji: '🦉', image: 'https://images.unsplash.com/photo-1568393691622-c7ba131d63b4?w=600&h=600&fit=crop' },
    { name: 'Zebra', emoji: '🦓', image: 'https://images.unsplash.com/photo-1551969014-7d2c4cddf0b6?w=600&h=600&fit=crop' },
    { name: 'Beer', emoji: '🐻', image: 'https://images.unsplash.com/photo-1589656966895-2f33e7653819?w=600&h=600&fit=crop' },
    { name: 'Eend', emoji: '🦆', image: 'https://images.unsplash.com/photo-1518384401463-7c0f195d5ce1?w=600&h=600&fit=crop' },
    { name: 'Koe', emoji: '🐮', image: 'https://images.unsplash.com/photo-1516467508483-a7212febe31a?w=600&h=600&fit=crop' },
    { name: 'Dolfijn', emoji: '🐬', image: 'https://images.unsplash.com/photo-1607153333879-c174d265f1d2?w=600&h=600&fit=crop' }
];

// Select random animal for this game
let currentAnimal = animals[Math.floor(Math.random() * animals.length)];
let imageLoaded = false;

// DOM elements
const numbersGrid = document.getElementById('numbersGrid');
const message = document.getElementById('message');
const resetBtn = document.getElementById('resetBtn');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const animalCover = document.getElementById('animalCover');
const animalImage = document.getElementById('animalImage');

// Initialize game
function initGame() {
    currentNumber = 1;
    correctAnswers = 0;
    currentAnimal = animals[Math.floor(Math.random() * animals.length)];
    imageLoaded = false;
    
    // Clear previous image
    animalImage.innerHTML = '';
    
    // Show loading state
    animalImage.innerHTML = '<div class="loading">Laden... ' + currentAnimal.emoji + '</div>';
    
    // Preload the animal image
    const img = new Image();
    img.onload = function() {
        imageLoaded = true;
        animalImage.innerHTML = '';
        animalImage.appendChild(img);
    };
    img.onerror = function() {
        // Fallback to emoji if image fails to load
        animalImage.innerHTML = '<div class="animal-emoji">' + currentAnimal.emoji + '</div>';
        imageLoaded = true;
    };
    img.src = currentAnimal.image;
    img.alt = currentAnimal.name;
    
    // Reset cover
    animalCover.style.clipPath = 'inset(0 0 0 0)';
    
    // Reset progress
    updateProgress();
    
    // Create number buttons
    createNumberButtons();
    
    // Clear message
    message.textContent = '';
    message.className = 'message';
}

// Create shuffled number buttons
function createNumberButtons() {
    numbersGrid.innerHTML = '';
    
    // Create array of numbers 1-10
    const numbers = Array.from({ length: totalNumbers }, (_, i) => i + 1);
    
    // Shuffle the array
    shuffleArray(numbers);
    
    // Create buttons
    numbers.forEach(num => {
        const button = document.createElement('button');
        button.className = 'number-btn';
        button.textContent = num;
        button.dataset.number = num;
        button.addEventListener('click', handleNumberClick);
        numbersGrid.appendChild(button);
    });
}

// Shuffle array function (Fisher-Yates algorithm)
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

// Handle number button click
function handleNumberClick(event) {
    const button = event.target;
    const clickedNumber = parseInt(button.dataset.number);
    
    if (clickedNumber === currentNumber) {
        // Correct answer!
        handleCorrectAnswer(button);
    } else {
        // Wrong answer
        handleWrongAnswer(button);
    }
}

// Handle correct answer
function handleCorrectAnswer(button) {
    correctAnswers++;
    currentNumber++;
    
    // Visual feedback
    button.classList.add('correct');
    button.disabled = true;
    
    // Update progress
    updateProgress();
    
    // Reveal more of the animal
    revealAnimal();
    
    // Show message
    if (correctAnswers === totalNumbers) {
        message.textContent = '🎉 Fantastisch! Je hebt alle getallen gevonden! 🎉';
        message.className = 'message success';
        disableAllButtons();
    } else {
        message.textContent = `✅ Goed gedaan! Zoek nu ${currentNumber}!`;
        message.className = 'message success';
    }
}

// Handle wrong answer
function handleWrongAnswer(button) {
    // Visual feedback
    button.classList.add('wrong');
    
    // Remove animation class after animation
    setTimeout(() => {
        button.classList.remove('wrong');
    }, 500);
    
    // Show message
    message.textContent = `❌ Oeps! Probeer nog eens. Zoek ${currentNumber}!`;
    message.className = 'message error';
}

// Update progress bar and text
function updateProgress() {
    const percentage = (correctAnswers / totalNumbers) * 100;
    progressBar.style.width = percentage + '%';
    progressText.textContent = `${correctAnswers}/${totalNumbers}`;
}

// Reveal animal progressively
function revealAnimal() {
    const percentage = (correctAnswers / totalNumbers) * 100;
    
    // Reveal from bottom to top - clip away percentage from bottom of cover
    // This keeps the cover visible at top, revealing image from bottom
    animalCover.style.clipPath = `inset(0 0 ${percentage}% 0)`;
}

// Disable all buttons
function disableAllButtons() {
    const buttons = document.querySelectorAll('.number-btn');
    buttons.forEach(button => {
        button.disabled = true;
    });
}

// Reset button handler
resetBtn.addEventListener('click', () => {
    initGame();
});

// Initialize game on load
initGame();
