// Game state
let currentNumber = 1;
let correctAnswers = 0;
const totalNumbers = 10;

// Array of animal emojis
const animals = ['🐶', '🐱', '🐼', '🦁', '🐯', '🐸', '🐰', '🦊', '🐻', '🐨', '🐷', '🐮', '🦒', '🐘', '🦓', '🦘', '🐧', '🦉', '🦆', '🐔'];

// Select random animal for this game
let currentAnimal = animals[Math.floor(Math.random() * animals.length)];

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
    
    // Display the animal
    animalImage.textContent = currentAnimal;
    
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
    
    // Reveal from bottom to top
    const revealAmount = 100 - percentage;
    animalCover.style.clipPath = `inset(0 0 ${revealAmount}% 0)`;
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
