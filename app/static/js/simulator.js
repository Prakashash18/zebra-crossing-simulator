// Simulator Game Logic
$(document).ready(function() {
    try {
        console.log("Document ready, initializing...");
        
        // Set up DOM references - using the existing constant references
        // This will ensure we're not trying to assign to constants
        
        // Add the awaiting-person class initially on load
        $("#cameraContainer").addClass("awaiting-person");
        
        // Force immediate voice initialization
        console.log("Initializing speech synthesis...");
        if (window.speechSynthesis) {
            // Force voices to load right away
            speechSynthesis.getVoices();
            loadVoices();
            
            // Set up the voices changed event
            if (speechSynthesis.onvoiceschanged !== undefined) {
                speechSynthesis.onvoiceschanged = function() {
                    console.log("Voices changed event fired");
                    loadVoices();
                    
                    // Pre-warm the speech engine with a silent utterance
                    if (selectedVoice) {
                        const warmUpUtterance = new SpeechSynthesisUtterance('');
                        warmUpUtterance.voice = selectedVoice;
                        speechSynthesis.speak(warmUpUtterance);
                    }
                };
            }
            
            // Try to pre-initialize with a silent utterance regardless
            const initUtterance = new SpeechSynthesisUtterance('');
            speechSynthesis.speak(initUtterance);
        } else {
            console.warn("Speech synthesis not available in this browser");
        }
        
        // Initialize audio controls
        initAudioControls();
        
        // Initialize game state (but don't speak)
        initializeGameState();
        
        // Initialize character
        renderCharacter();
        
        // Setup button handlers - ONLY set these up once
        $startButton.off('click').on('click', function() {
            console.log("Start button clicked");
            startGame();
        });
        
        $resetButton.off('click').on('click', function() {
            console.log("Reset button clicked");
            resetGame(true); // true = is actually resetting
        });
        
        // Start the initial setup flow for first time
        initialSetup();
        
        console.log("Initialization complete");
    } catch (error) {
        console.error("Error during initialization:", error);
    }
});

// Initialize game state without speaking
function initializeGameState() {
    gameState.started = false;
    gameState.currentStep = 0;
    gameState.points = 0;
    gameState.correctPoses = 0;
    gameState.waitingForPose = false;
    
    resetCharacter();
    speechSynthesis.cancel();
    stopBackgroundMusic();
    stopTrafficSound();
    updateUIState();
    
    console.log('Game state initialized');
}

// Game state
const gameState = {
    started: false,
    currentStep: 0,
    points: 0,
    level: 1,
    correctPoses: 0,
    totalCrossings: 0,
    waitingForPose: false,
    currentDetectedPose: null,
    initialized: false,
    stepStartTime: null,
    stepTimeLimit: 5000, // 5 seconds to complete each step
    stepTimerInterval: null, // Track the timer interval ID
    retryCount: 0,
    maxRetries: 3,
    currentPose: null,
    steps: [
        { pose: "Looking Straight", instruction: "Great! Let's begin. First, look straight ahead." },
        { pose: "Looking Left", instruction: "Now, look to your left to check for traffic." },
        { pose: "Looking Right", instruction: "Good! Now look to your right to check for traffic again." },
        { pose: "Looking Straight", instruction: "Now look straight ahead again." },
        { pose: "Hand Up", instruction: "Now raise your hand to signal that you want to cross." },
        { pose: "Walking", instruction: "Great! Now you can cross the road safely. Walk forward!" }
    ],
    poseBuffer: {},
    lastPoseTime: 0,
    personDetected: false,
    lookingStraight: false,
    initializing: true,
    connectionAttempts: 0,
    maxConnectionAttempts: 5,
    reconnectDelay: 1000,
    isReconnecting: false,
    lastSuccessfulConnection: 0,
    
    // Speech tracking flags
    greetingSpeechComplete: false,
    lookStraightSpeechStarted: false,
    lookStraightSpeechComplete: false,
    startButtonSpeechComplete: false,
    startButtonSpeechInProgress: false,
    isGreetingInProgress: false
};

// UI elements
const $currentPose = $("#currentPose");
const $poseToPerform = $("#poseToPerform");
const $currentStep = $("#currentStep");
const $instructionText = $("#instructionText");
const $points = $("#points");
const $level = $("#level");
const $progressBar = $("#progressBar");
const $startButton = $("#startButton");
const $resetButton = $("#resetButton");
const $countdownContainer = $("#countdownContainer");
const $countdown = $("#countdown");
const $character = $("#character");
const $initialStatus = $("#instructionText"); // Use instruction text for initial status
const $stepTimer = $("#stepTimer"); // Timer indicator
const $successFeedback = $("#successFeedback"); // Success feedback
const $failureFeedback = $("#failureFeedback"); // Failure feedback
const $timeoutFeedback = $("#timeoutFeedback"); // Timeout feedback
const $instructionsContainer = $("#instructionsContainer"); // Instructions container
const $messageDiv = $(".message");

// Game configuration
const config = {
    pollInterval: 300,
    poseConfidenceThreshold: 0.4,
    bufferThreshold: 0.5,
    requiredDuration: 1000, // 1 second
    countdownDuration: 3,
    timeoutPenalty: 5, // Points deducted for timeout
    gameSteps: [
        {
            name: "Look Left",
            instruction: "Look to your left",
            detailedInstruction: "Turn your head to the left and check for any vehicles coming from that direction",
            poseRequired: "Looking Left",
            points: 10,
            successMessage: "Great job looking left! 👀",
            failureMessage: "Let's try looking left again.",
            timeoutMessage: "You didn't look left in time. -5 points. Remember to look left before crossing!",
            characterAction: "look-left",
            emoji: "👈"
        },
        {
            name: "Look Right",
            instruction: "Look to your right",
            detailedInstruction: "Now turn your head to the right and check for any vehicles coming from that direction",
            poseRequired: "Looking Right",
            points: 10,
            successMessage: "Excellent! You checked the right side! 👀",
            failureMessage: "Let's practice looking right again.",
            timeoutMessage: "You didn't look right in time. -5 points. Remember to look right to see any cars!",
            characterAction: "look-right",
            emoji: "👉"
        },
        {
            name: "Look Left Again",
            instruction: "Look left one more time",
            detailedInstruction: "One final check to the left to make sure it's safe to cross",
            poseRequired: "Looking Left",
            points: 10,
            successMessage: "Perfect! Final check complete! 👀",
            failureMessage: "Let's do one more check to the left.",
            timeoutMessage: "You didn't look left again in time. -5 points. Always check left one more time before crossing!",
            characterAction: "look-left",
            emoji: "👈"
        },
        {
            name: "Signal to Cross",
            instruction: "Raise your right hand",
            detailedInstruction: "Raise your right hand high to let drivers know you want to cross",
            poseRequired: "Raise Right Hand",
            points: 15,
            successMessage: "Excellent signaling! ✋",
            failureMessage: "Let's try raising your right hand again.",
            timeoutMessage: "You didn't signal in time. -5 points. Remember to signal before crossing!",
            characterAction: "raise-right-hand",
            emoji: "✋"
        }
    ]
};

const INITIAL_POSE = "Looking Straight";
const INIT_CHECK_INTERVAL = 500; // Check every 500ms for initial pose

// Initialize speech synthesis
let speechSynthesis = window.speechSynthesis;
let availableVoices = [];
let selectedVoice = null;

// Audio state
let isMusicEnabled = true;
let isSoundEnabled = true;

// Audio elements
const backgroundMusic = document.getElementById('backgroundMusic');
const trafficSound = document.getElementById('trafficSound');
const successSound = document.getElementById('successSound');
const countdownSound = document.getElementById('countdownSound');
const failureSound = document.getElementById('failureSound'); // Define failureSound

// Audio control buttons
const toggleMusicBtn = document.getElementById('toggleMusic');
const toggleSoundBtn = document.getElementById('toggleSound');

// Speech queue to manage multiple speech requests
let speechQueue = [];
let isSpeaking = false;
let currentSpeechPromise = null;

// Load voices for speech synthesis
function loadVoices() {
    availableVoices = speechSynthesis.getVoices();
    console.log("Voices loaded:", availableVoices.length);
    
    // ONLY look for female voices - prioritize specifically by name
    const femaleVoiceNames = [
        'samantha', 'karen', 'victoria', 'lisa', 'amy', 'allison', 'susan', 'kathy',
        'monica', 'catherine', 'fiona', 'veena', 'tessa', 'female'
    ];
    
    // First try to find specific named female voices
    for (const name of femaleVoiceNames) {
        const matchedVoice = availableVoices.find(voice => 
            voice.name.toLowerCase().includes(name)
        );
        if (matchedVoice) {
            selectedVoice = matchedVoice;
            console.log(`Selected specific female voice: ${selectedVoice.name}`);
            break;
        }
    }
    
    // If no specific female voice found, look for any voice with 'female' in the name or description
    if (!selectedVoice) {
        selectedVoice = availableVoices.find(voice => 
            voice.name.toLowerCase().includes('female') || 
            (voice.name.toLowerCase().includes('en') && voice.name.toLowerCase().includes('female'))
        );
        
        if (selectedVoice) {
            console.log(`Selected generic female voice: ${selectedVoice.name}`);
        }
    }
    
    // Third fallback: any English voice that might be female
    if (!selectedVoice) {
        selectedVoice = availableVoices.find(voice => 
            (voice.name.toLowerCase().includes('en') || voice.lang.startsWith('en-')) &&
            !voice.name.toLowerCase().includes('male') &&
            !voice.name.toLowerCase().includes('man')
        );
        
        if (selectedVoice) {
            console.log(`Selected potential female English voice: ${selectedVoice.name}`);
        }
    }
    
    // Last resort: just use any voice
    if (!selectedVoice && availableVoices.length > 0) {
        selectedVoice = availableVoices[0];
        console.warn(`No female voice found, using default: ${selectedVoice.name}`);
    }
    
    if (selectedVoice) {
        console.log("Selected voice:", selectedVoice.name);
        // Test the voice immediately with a silent utterance to "warm up" the engine
        const testUtterance = new SpeechSynthesisUtterance('');
        testUtterance.voice = selectedVoice;
        speechSynthesis.speak(testUtterance);
    } else {
        console.error("No voices available at all");
    }
}

// Improve the speech queue system to prevent interruptions
function processSpeechQueue() {
    if (speechQueue.length === 0 || isSpeaking) return;
    
    isSpeaking = true;
    const speechItem = speechQueue.shift();
    const { text, resolve, reject } = speechItem;
    
    try {
        console.log("Processing speech from queue:", text);
        
        if (!window.speechSynthesis) {
            console.warn("Speech synthesis not available");
            isSpeaking = false;
            resolve(); // Resolve anyway so game can continue
            setTimeout(processSpeechQueue, 100);
            return;
        }
        
        // Stop any current speech to avoid conflicts
        speechSynthesis.cancel();
        
        // Create a new utterance for the text
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Add event handlers for debugging
        utterance.onstart = function() {
            console.log("Speech started:", text.substring(0, 20) + "...");
            // Show visual indicator that speech is happening
            const $speechIndicator = $('<div class="speech-indicator"><i class="fas fa-volume-up fa-pulse"></i></div>');
            $('body').append($speechIndicator);
            // Store it with the utterance
            utterance.indicator = $speechIndicator;
        };
        
        // Use the selected voice with natural settings for a female voice
        if (selectedVoice) {
            console.log("Using voice:", selectedVoice.name);
            utterance.voice = selectedVoice;
            utterance.rate = 0.95;  // Slightly slower for clarity
            utterance.pitch = 1.2;  // Slightly higher pitch for female voice
            utterance.volume = 1.0; // Full volume
        } else {
            console.warn("No voice selected, trying to force load voices again");
            loadVoices();
            
            // If we found a voice now, use it
            if (selectedVoice) {
                utterance.voice = selectedVoice;
                utterance.pitch = 1.2; // Slightly higher pitch for female voice
                utterance.rate = 0.95; // Slightly slower for clarity
            } else {
                // Still no voice, try to use system defaults that might be feminine
                utterance.pitch = 1.2; // Higher pitch as fallback to make more female-like
                utterance.rate = 0.95; // Slightly slower for clarity
            }
        }
        
        // Remove indicator when speech ends and process next in queue
        utterance.onend = function() {
            console.log("Speech completed successfully:", text.substring(0, 20) + "...");
            if (utterance.indicator) {
                utterance.indicator.fadeOut(300, function() {
                    $(this).remove();
                });
            }
            isSpeaking = false;
            
            // Wait a moment before processing the next speech
            // This adds a natural pause between sentences
            setTimeout(() => {
                resolve(); // Resolve the promise
                setTimeout(processSpeechQueue, 300); // Small delay between speeches
            }, 200);
        };
        
        // Also remove on error and continue with queue
        utterance.onerror = function(event) {
            console.error("Speech error:", event.error);
            if (utterance.indicator) {
                utterance.indicator.fadeOut(300, function() {
                    $(this).remove();
                });
            }
            isSpeaking = false;
            
            // Try to diagnose the issue
            let errorInfo = "Unknown speech error";
            if (event.error === 'interrupted') {
                errorInfo = "Speech was interrupted by another utterance";
            } else if (event.error === 'canceled') {
                errorInfo = "Speech was canceled";
            } else if (event.error === 'network') {
                errorInfo = "Network error occurred during speech synthesis";
            }
            console.warn(errorInfo);
            
            setTimeout(() => {
                resolve(); // Resolve anyway so game can continue
                setTimeout(processSpeechQueue, 300);
            }, 200);
        };
        
        // Actually speak the text
        speechSynthesis.speak(utterance);
        
        // Set a safety timeout to remove the indicator and continue queue if speech doesn't complete
        setTimeout(function() {
            if (isSpeaking) {
                console.warn("Speech indicator timeout - cleaning up");
                if (utterance.indicator) {
                    utterance.indicator.fadeOut(300, function() {
                        $(this).remove();
                    });
                }
                isSpeaking = false;
                resolve(); // Resolve anyway after timeout
                setTimeout(processSpeechQueue, 300);
            }
        }, 15000); // 15 seconds max (increased from 10)
        
    } catch (error) {
        console.error("Error in speech synthesis:", error);
        isSpeaking = false;
        reject(error);
        setTimeout(processSpeechQueue, 300);
    }
}

// Enhanced speak function to manage speech better
function speak(text) {
    try {
        if (!text || typeof text !== 'string') {
            console.warn("Invalid text for speech:", text);
            return Promise.resolve(); // Return resolved promise for empty text
        }
        
        console.log("Adding to speech queue:", text);
        
        // If no voices are loaded yet, try to load them now
        if (!selectedVoice && speechSynthesis) {
            console.log("No voice selected yet, trying to load voices");
            loadVoices();
        }
        
        // Create a new promise for this speech request
        const speechPromise = new Promise((resolve, reject) => {
            // Add text to queue with its resolver
            speechQueue.push({ text, resolve, reject });
        });
        
        // Start processing the queue if not already speaking
        if (!isSpeaking) {
            processSpeechQueue();
        }
        
        // Return the promise that will resolve when this speech completes
        return speechPromise;
        
    } catch (error) {
        console.error("Error queueing speech:", error);
        return Promise.reject(error);
    }
}

// Initialize audio controls
function initAudioControls() {
    // Set initial button states
    updateAudioButtonStates();
    
    // Add event listeners for audio controls
    toggleMusicBtn.addEventListener('click', () => {
        isMusicEnabled = !isMusicEnabled;
        updateAudioButtonStates();
        if (isMusicEnabled) {
            backgroundMusic.play().catch(err => console.log('Background music play failed:', err));
        } else {
            backgroundMusic.pause();
        }
    });

    toggleSoundBtn.addEventListener('click', () => {
        isSoundEnabled = !isSoundEnabled;
        updateAudioButtonStates();
    });
}

// Update audio button states
function updateAudioButtonStates() {
    toggleMusicBtn.innerHTML = `<i class="fas fa-music"></i> Music ${isMusicEnabled ? 'ON' : 'OFF'}`;
    toggleSoundBtn.innerHTML = `<i class="fas fa-volume-up"></i> Sound ${isSoundEnabled ? 'ON' : 'OFF'}`;
}

// Play sound effect
function playSound(sound) {
    if (!isSoundEnabled) return;
    
    sound.currentTime = 0;
    sound.play().catch(err => console.log('Sound play failed:', err));
}

// Start the game
function startGame() {
    try {
        console.log("Start game called, game state:", gameState.started);
        
        // Log the DOM element that was clicked to help debug
        console.log("Start button DOM element:", $startButton[0]);
        
        if (gameState.started) {
            console.log("Game already started, ignoring startGame call");
            return;
        }
        
        // Make sure the start button speech is complete before proceeding
        if (gameState.startButtonSpeechInProgress && !gameState.startButtonSpeechComplete) {
            console.log("Start button speech not complete yet, waiting...");
            setTimeout(startGame, 500);
            return;
        }
        
        console.log("Starting game initialization sequence");
        
        // Disable start button during countdown
        $startButton.prop('disabled', true)
            .css('opacity', 0.7)
            .text('Starting...');
        
        // Play countdown sound
        if (countdownSound) {
            console.log("Playing countdown sound");
            playSound(countdownSound);
        } else {
            console.warn("Countdown sound not available");
        }
        
        // Show countdown
        const countdownContainer = document.getElementById('countdownContainer');
        const countdownElement = document.getElementById('countdown');
        
        if (!countdownContainer || !countdownElement) {
            console.error("Countdown elements not found in DOM");
            return;
        }
        
        countdownContainer.classList.remove('d-none');
        
        let count = 3;
        countdownElement.textContent = count;
        console.log("Countdown started:", count);
        
        // Speak the countdown and proceed only when speech is done
        runCountdown(count, countdownContainer, countdownElement);
        
    } catch (error) {
        console.error('Error in startGame:', error);
        // Try to recover gracefully
        alert("There was an error starting the game. Please refresh the page and try again.");
        resetGame();
    }
}

// Run countdown with speech promises
function runCountdown(count, countdownContainer, countdownElement) {
    // First announcement before countdown
    speak("Let's get ready!")
        .then(() => {
            // After initial announcement, start the actual countdown
            countdownElement.textContent = count;
            
            // Play sound effect
            if (successSound) {
                playSound(successSound);
            }
            
            // Create a recursive function for counting down
            function countDown(currentCount) {
                // Speak the current number
                speak(currentCount.toString())
                    .then(() => {
                        // After speech completes, decrement count
                        currentCount--;
                        
                        if (currentCount > 0) {
                            console.log("Countdown:", currentCount);
                            countdownElement.textContent = currentCount;
                            
                            // Play sound effect
                            if (successSound) {
                                playSound(successSound);
                            }
                            
                            // Continue countdown
                            countDown(currentCount);
                        } else {
                            // Countdown complete
                            countdownContainer.classList.add('d-none');
                            console.log("Countdown complete, game starting now");
                            
                            // Initialize game state
                            gameState.started = true;
                            gameState.currentStep = 0;
                            gameState.points = 0;
                            gameState.correctPoses = 0;
                            gameState.waitingForPose = false; // Start as false until instructions are spoken
                            
                            console.log("Game fully started, waiting for pose detection");
                            
                            // Reset character and UI
                            resetCharacter();
                            updateUIState();
                            
                            // Ensure camera container doesn't have awaiting-person class during game
                            $("#cameraContainer").removeClass("awaiting-person");
                            
                            // Start traffic sound
                            console.log("Starting traffic sound");
                            startTrafficSound();
                            
                            // Show first instruction
                            console.log("Showing first step instructions");
                            showCurrentStepInstructions();
                            
                            // Final announcement
                            speak("Let's begin!");
                        }
                    })
                    .catch(error => {
                        console.error("Error during countdown speech:", error);
                        // Try to recover by continuing countdown
                        proceedAfterError(currentCount);
                    });
            }
            
            // Start the countdown
            countDown(count);
        })
        .catch(error => {
            console.error("Error during initial countdown announcement:", error);
            // Try to recover
            proceedAfterError(count);
        });
    
    // Helper function to proceed after error
    function proceedAfterError(currentCount) {
        if (currentCount > 1) {
            countdownElement.textContent = currentCount - 1;
            countDown(currentCount - 1);
        } else {
            // If at end of countdown, just start game
            countdownContainer.classList.add('d-none');
            
            // Initialize game state
            gameState.started = true;
            gameState.currentStep = 0;
            gameState.points = 0;
            gameState.correctPoses = 0;
            gameState.waitingForPose = false;
            
            // Reset character and UI
            resetCharacter();
            updateUIState();
            
            // Remove awaiting-person class
            $("#cameraContainer").removeClass("awaiting-person");
            
            // Start traffic sound
            startTrafficSound();
            
            // Show first instruction
            showCurrentStepInstructions();
        }
    }
}

// Reset game
function resetGame(isActualReset = false) {
    try {
        console.log("Game reset called, isActualReset:", isActualReset);
        
        // Clear any running step timer
        if (gameState.stepTimerInterval) {
            clearInterval(gameState.stepTimerInterval);
            gameState.stepTimerInterval = null;
        }
        
        // Reset all game state flags
        gameState.started = false;
        gameState.currentStep = 0;
        gameState.points = 0;
        gameState.correctPoses = 0;
        gameState.waitingForPose = false;
        gameState.retryCount = 0;
        gameState.startButtonSpeechComplete = false;
        gameState.startButtonSpeechInProgress = false;
        
        // Clear any speech and audio
        speechSynthesis.cancel();
        stopBackgroundMusic();
        stopTrafficSound();
        
        // Clear animation classes from UI elements
        $character.removeClass('animate__animated animate__bounce animate__tada');
        $currentStep.removeClass('animate__animated animate__bounceIn animate__tada');
        $instructionText.removeClass('animate__animated animate__fadeIn');
        $poseToPerform.parent().removeClass('animate__animated animate__pulse alert-danger alert-warning alert-success');
        $points.removeClass('animate__animated animate__heartBeat text-danger');
        
        // Reset visual elements
        resetCharacter();
        
        // Hide the countdown container
        $("#countdownContainer").addClass("d-none");
        
        // Remove any feedback elements
        $successFeedback.css('opacity', 0);
        $failureFeedback.css('opacity', 0);
        $timeoutFeedback.css('opacity', 0);
        
        // Reset the step timer
        $stepTimer.css({
            'width': '100%',
            'background-color': 'var(--primary-color)'
        }).removeClass('animate__animated animate__pulse animate__infinite').html('');
        
        // Update UI based on reset state
        updateUIState();
        
        // Reset button visibility
        $resetButton.addClass('d-none');
        $startButton.text('Start Exercise')
            .removeClass('d-none')
            .prop('disabled', true)
            .css('opacity', 1);
        
        // Add the awaiting-person class back when resetting
        $("#cameraContainer").addClass("awaiting-person");
        
        // Reset pose detection state for initialization
        gameState.initialized = false;
        gameState.personDetected = false;
        gameState.lookingStraight = false;
        gameState.greetingSpeechComplete = false;
        gameState.lookStraightSpeechComplete = false;
        gameState.lookStraightSpeechStarted = false;
        gameState.isGreetingInProgress = false;
        
        // Only speak a reset message if this is an actual reset (not initial setup)
        if (isActualReset) {
            console.log('Game reset complete, providing reset instructions');
            // Update instruction text while we wait for detection
            $instructionText.text("Please stand in front of the camera");
            $currentPose.text("Waiting for user...");
            $poseToPerform.text("Stand in front of the camera");
            $poseToPerform.parent()
                .removeClass('alert-success alert-danger')
                .addClass('alert-info');
                
            // Slight delay to ensure UI updates before speaking
            setTimeout(() => {
                speak("Let's try again! Please stand in front of the camera.").catch(err => {
                    console.error("Error during reset speech:", err);
                });
            }, 200);
        }
        
        // Start the detection process
        setTimeout(checkInitialPose, 500);
        
        console.log("Game has been reset, beginning new detection cycle");
    } catch (error) {
        console.error('Error in resetGame:', error);
        // Try to recover from errors during reset
        alert("There was an error resetting the game. Please refresh the page.");
    }
}

// Show current step instructions with animation
function showCurrentStepInstructions(isRetry = false) {
    try {
        const currentStep = config.gameSteps[gameState.currentStep];
        if (!currentStep) return;
        
        // Clear any previous speech
        speechSynthesis.cancel();
        
        // Prepare for animation by removing old classes
        $currentStep.removeClass('animate__animated animate__bounceIn');
        $instructionText.removeClass('animate__animated animate__fadeIn');
        $poseToPerform.parent().removeClass('animate__animated animate__pulse alert-danger alert-warning alert-success');
        
        // Reset feedback elements
        $successFeedback.css('opacity', 0);
        $failureFeedback.css('opacity', 0);
        $timeoutFeedback.css('opacity', 0);
        
        // Reset and start the step timer with clear styling
        $stepTimer.css({
            'width': '100%',
            'background-color': isRetry ? 'var(--secondary-color)' : 'var(--primary-color)'
        }).removeClass('animate__animated animate__pulse animate__infinite').html('');
        
        // Force DOM reflow to restart animation
        void $currentStep[0].offsetWidth;
        void $instructionText[0].offsetWidth;
        void $poseToPerform.parent()[0].offsetWidth;
        
        // Update UI elements with step emoji
        $currentStep.html(`${currentStep.emoji} ${currentStep.instruction}`)
            .addClass('animate__animated animate__bounceIn');
        
        // Only update detailed instructions and speak on first try or after timeout
        if (!isRetry) {
            $instructionText.text(currentStep.detailedInstruction)
                .addClass('animate__animated animate__fadeIn');
            
            // Speak instruction with slight delay to ensure UI is updated
            setTimeout(() => {
                // Wait for speech to complete before allowing pose detection
                speak(currentStep.detailedInstruction).then(() => {
                    console.log("Step instruction speech completed, starting pose detection");
                    
                    // Start the pose detection and timer ONLY after speech completes
                    gameState.waitingForPose = true;
                    
                    // Reset the step timer when we actually start checking for poses
                    gameState.stepStartTime = Date.now();
                    console.log(`Step timer started at ${gameState.stepStartTime}, limit is ${gameState.stepTimeLimit}ms`);
                    
                    // Start the timer visualization
                    startStepTimer();
                    
                    // Start pose detection
                    checkPose();
                }).catch(error => {
                    console.error("Error during instruction speech:", error);
                    // Still continue with pose detection even if speech fails
                    gameState.waitingForPose = true;
                    gameState.stepStartTime = Date.now();
                    startStepTimer();
                    checkPose();
                });
            }, 500);
        } else {
            // On retry, use the failure message instead
            $instructionText.text(currentStep.failureMessage)
                .addClass('animate__animated animate__fadeIn');
            
            // Add retry class to the game area
            $('#gameArea').addClass('step-retry');
            
            // Speak retry instruction
            setTimeout(() => {
                speak(currentStep.failureMessage).then(() => {
                    console.log("Retry instruction speech completed, continuing pose detection");
                    gameState.waitingForPose = true;
                    
                    // Reset the step timer when we actually start checking for poses on retry
                    gameState.stepStartTime = Date.now();
                    console.log(`Step timer restarted at ${gameState.stepStartTime} for retry`);
                    
                    // Start the timer visualization
                    startStepTimer();
                    
                    // Start pose detection
                    checkPose();
                }).catch(error => {
                    console.error("Error during retry instruction speech:", error);
                    gameState.waitingForPose = true;
                    gameState.stepStartTime = Date.now();
                    startStepTimer();
                    checkPose();
                });
            }, 500);
            
            // Remove retry class after a short delay
            setTimeout(() => {
                $('#gameArea').removeClass('step-retry');
            }, 1500);
        }
        
        // Update the pose instruction UI based on retry status
        if (isRetry) {
            $poseToPerform.text(`Try again: ${currentStep.instruction}`);
            $poseToPerform.parent()
                .addClass('alert-warning animate__animated animate__headShake');
        } else {
            $poseToPerform.text(currentStep.instruction);
            $poseToPerform.parent()
                .addClass('alert-info animate__animated animate__pulse');
        }
        
        // Update character to show the required pose
        updateCharacterPose(currentStep.characterAction);
        $character.addClass('character-animated');
        
        // Add a countdown text to let the user know they have 5 seconds
        $stepTimer.attr('title', '5 seconds to complete this pose');
        
        updateProgressBar();
    } catch (error) {
        console.error('Error in showCurrentStepInstructions:', error);
        // Recover by enabling pose detection anyway
        gameState.waitingForPose = true;
        gameState.stepStartTime = Date.now();
        startStepTimer();
    }
}

// New function to start the step timer with proper monitoring
function startStepTimer() {
    // Ensure any previous timer is cleared
    if (gameState.stepTimerInterval) {
        clearInterval(gameState.stepTimerInterval);
    }
    
    // Create a new interval to check timeout frequently
    gameState.stepTimerInterval = setInterval(checkStepTimeout, 100);
    
    console.log("Step timer started with interval ID:", gameState.stepTimerInterval);
}

// Check step timeout with visual countdown
function checkStepTimeout() {
    // Only check if we're waiting for a pose
    if (!gameState.waitingForPose || !gameState.started) {
        // If we're not waiting, clear the interval
        if (gameState.stepTimerInterval) {
            console.log("Clearing step timer interval because no longer waiting for pose");
            clearInterval(gameState.stepTimerInterval);
            gameState.stepTimerInterval = null;
        }
        return;
    }
    
    const currentTime = Date.now();
    const elapsedTime = currentTime - gameState.stepStartTime;
    const timeRemaining = Math.max(0, gameState.stepTimeLimit - elapsedTime);
    
    // Debug log every second to track timer progress
    if (Math.floor(elapsedTime / 1000) !== Math.floor((elapsedTime - 100) / 1000)) {
        console.log(`Step timer: ${elapsedTime}ms elapsed, ${timeRemaining}ms remaining`);
    }
    
    // Update the timer indicator
    const percentRemaining = (timeRemaining / gameState.stepTimeLimit) * 100;
    $stepTimer.css('width', `${percentRemaining}%`);
    
    // Change color as time runs low with more distinctive transitions
    if (percentRemaining < 20) {
        $stepTimer.css('background-color', '#ff3b30'); // Bright red for urgent warning
        // Add pulse animation for final warning
        if (!$stepTimer.hasClass('animate__pulse') && percentRemaining < 10) {
            $stepTimer.addClass('animate__animated animate__pulse animate__infinite');
        }
    } else if (percentRemaining < 50) {
        $stepTimer.css('background-color', '#ff9500'); // Orange for warning
        $stepTimer.removeClass('animate__animated animate__pulse animate__infinite');
    } else {
        $stepTimer.css('background-color', 'var(--primary-color)'); // Default
        $stepTimer.removeClass('animate__animated animate__pulse animate__infinite');
    }
    
    // Show time remaining in seconds if under 3 seconds
    if (timeRemaining < 3000 && timeRemaining > 0) {
        const secondsLeft = Math.ceil(timeRemaining / 1000);
        if (!$stepTimer.attr('data-seconds') || $stepTimer.attr('data-seconds') != secondsLeft) {
            $stepTimer.attr('data-seconds', secondsLeft);
            $stepTimer.html(`<span class="small">${secondsLeft}s</span>`);
        }
    } else {
        $stepTimer.html('');
    }
    
    // If time limit exceeded
    if (elapsedTime >= gameState.stepTimeLimit) {
        console.log(`TIMEOUT: Step time limit of ${gameState.stepTimeLimit}ms exceeded (elapsed: ${elapsedTime}ms)`);
        
        // Clear the interval
        if (gameState.stepTimerInterval) {
            clearInterval(gameState.stepTimerInterval);
            gameState.stepTimerInterval = null;
        }
        
        // Clear any timer visuals
        $stepTimer.removeClass('animate__animated animate__pulse animate__infinite');
        
        // Call timeout handler
        handleStepTimeout();
    }
}

// Handle step timeout with enhanced feedback and point deduction
function handleStepTimeout() {
    if (!gameState.waitingForPose) return;
    
    gameState.waitingForPose = false; // Stop checking while handling timeout
    const currentStep = config.gameSteps[gameState.currentStep];
    
    console.log("Handling timeout for step:", currentStep.name);
    
    // Show timeout feedback with more prominent visual cue
    $timeoutFeedback.css('opacity', 1);
    setTimeout(() => {
        $timeoutFeedback.css('opacity', 0);
    }, 2000);
    
    // Deduct points for timeout
    const pointsLost = config.timeoutPenalty;
    gameState.points = Math.max(0, gameState.points - pointsLost); // Ensure points don't go below zero
    
    // Create a visual indicator for point deduction
    const $pointDeduction = $('<div class="point-deduction">-' + pointsLost + '</div>');
    $('#gameArea').append($pointDeduction);
    
    // Animate the point deduction indicator
    $pointDeduction.animate({
        top: '-=50px',
        opacity: 0
    }, 1500, function() {
        $(this).remove();
    });
    
    // Update the points display with animation
    $points.text(`${gameState.points}/${MAX_POINTS}`)
        .addClass('animate__animated animate__flash text-danger');
    
    // Remove animation classes after animation completes
    setTimeout(() => {
        $points.removeClass('animate__animated animate__flash text-danger');
    }, 1500);
    
    // Indicate timeout to the user
    $poseToPerform.parent()
        .removeClass('alert-info alert-warning alert-success')
        .addClass('alert-danger animate__animated animate__headShake');
    
    $poseToPerform.html(`<i class="fas fa-exclamation-triangle"></i> ${currentStep.timeoutMessage}`);
    
    // Character shows disappointment
    $character.addClass('character-timeout');
    setTimeout(() => {
        $character.removeClass('character-timeout');
    }, 1500);
    
    // Increase retry count
    gameState.retryCount++;
    
    // Play an error sound
    if (isSoundEnabled) {
        // Safely check if failureSound exists before using it
        if (failureSound && typeof failureSound.play === 'function') {
            playSound(failureSound);
        } else if (successSound) {
            // If no dedicated failure sound, use success sound at lower pitch as fallback
            const tempSound = successSound.cloneNode();
            tempSound.playbackRate = 0.7; // Slower/lower pitch for negative feedback
            tempSound.volume = 0.6;
            tempSound.play().catch(err => console.log('Sound play failed:', err));
        }
    }
    
    // Speak the timeout message and wait for completion
    speak(currentStep.timeoutMessage)
        .then(() => {
            console.log("Timeout message speech completed");
            
            // After speech is complete, retry or move on if too many retries
            if (gameState.retryCount >= gameState.maxRetries) {
                // Too many retries, move to next step anyway
                console.log("Max retries reached, moving to next step");
                gameState.retryCount = 0;
                moveToNextStep();
            } else {
                // Retry the current step
                console.log(`Retry attempt ${gameState.retryCount} of ${gameState.maxRetries}`);
                showCurrentStepInstructions(true); // true = retry
            }
        })
        .catch(error => {
            console.error("Error during timeout message speech:", error);
            
            // Even if speech fails, still continue with game flow
            if (gameState.retryCount >= gameState.maxRetries) {
                gameState.retryCount = 0;
                moveToNextStep();
            } else {
                showCurrentStepInstructions(true);
            }
        });
}

// Handle pose detection success
function handleCorrectPose() {
    try {
        if (!gameState.waitingForPose) return;
        
        console.log("Handling correct pose");
        gameState.waitingForPose = false;
        
        // Clear the step timer interval since we got the correct pose
        if (gameState.stepTimerInterval) {
            clearInterval(gameState.stepTimerInterval);
            gameState.stepTimerInterval = null;
        }
        
        gameState.correctPoses++;
        gameState.retryCount = 0; // Reset retry count on success
        
        // Show success feedback
        $successFeedback.css('opacity', 1);
        setTimeout(() => {
            $successFeedback.css('opacity', 0);
        }, 1000);
        
        const currentStep = config.gameSteps[gameState.currentStep];
        
        // Play success sound
        playSound(successSound);
        
        // Celebration animation for success
        $poseToPerform.parent()
            .removeClass('alert-info alert-warning alert-danger animate__pulse animate__headShake')
            .addClass('alert-success animate__animated animate__bounceIn');
        
        $poseToPerform.html(`${currentStep.successMessage} <span class="fs-2">🎉</span>`);
        
        // Add points with animation
        gameState.points += currentStep.points;
        $points.text(`${gameState.points}/${MAX_POINTS}`)
            .addClass('animate__animated animate__heartBeat');
        
        // Stop the timer
        $stepTimer.css('width', '100%').css('background-color', 'var(--success-color)');
        
        // Add character success animation
        $character.removeClass('character-animated').addClass('animate__animated animate__bounce');
        
        // Remove animation class after it completes
        setTimeout(() => {
            $points.removeClass('animate__animated animate__heartBeat');
            $character.removeClass('animate__animated animate__bounce');
        }, 1000);
        
        // First speak the success message and WAIT for it to complete
        speak("Well done!").then(() => {
            console.log("Success speech completed, moving to next step");
            // Only move to next step after speech is complete
            moveToNextStep();
        }).catch(error => {
            console.error("Error during success speech:", error);
            // Still move to next step even if speech fails
            moveToNextStep();
        });
        
    } catch (error) {
        console.error('Error in handleCorrectPose:', error);
    }
}

// Move to next step with animation
function moveToNextStep() {
    try {
        console.log("Moving to next step");
        
        // Make sure instructionsContainer exists
        if (!$instructionsContainer || $instructionsContainer.length === 0) {
            console.error("Instructions container not found:", $instructionsContainer);
            $instructionsContainer = $("#instructionsContainer");
            
            if (!$instructionsContainer || $instructionsContainer.length === 0) {
                console.error("Still cannot find instructions container, continuing without animation");
                proceedToNextStep();
                return;
            }
        }
        
        // Add exit animation to current step content
        $instructionsContainer.addClass('animate__animated animate__fadeOutUp');
        
        // Wait for exit animation to complete
        setTimeout(() => {
            proceedToNextStep();
        }, 500); // Time for exit animation
        
    } catch (error) {
        console.error('Error in moveToNextStep:', error);
        // Try to continue without animation if there's an error
        proceedToNextStep();
    }
}

// Actually proceed to the next step after animations
function proceedToNextStep() {
    try {
        gameState.currentStep++;
        
        // Remove animations from instructions container
        if ($instructionsContainer && $instructionsContainer.length > 0) {
            $instructionsContainer.removeClass('animate__animated animate__fadeOutUp');
        }
        
        if (gameState.currentStep >= config.gameSteps.length) {
            completeGame();
        } else {
            // Add entrance animation
            if ($instructionsContainer && $instructionsContainer.length > 0) {
                $instructionsContainer.addClass('animate__animated animate__fadeInDown');
                
                // Remove entrance animation after it completes
                setTimeout(() => {
                    $instructionsContainer.removeClass('animate__animated animate__fadeInDown');
                }, 1000);
            }
            
            // Show the new step
            showCurrentStepInstructions(false); // New step, not a retry
            
            console.log("Now at step", gameState.currentStep, "restarting pose check.");
            checkPose();
        }
    } catch (error) {
        console.error('Error in proceedToNextStep:', error);
    }
}

// Complete game
function completeGame() {
    gameState.started = false;
    gameState.totalCrossings++;
    
    // Move character to end of crossing with animation
    moveCharacterForward(100);
    $progressBar.css('width', '100%');
    
    // Show celebration with animation
    $instructionsContainer
        .removeClass('animate__animated animate__fadeOutUp')
        .addClass('animate__animated animate__bounceIn');
    
    $currentStep.html("🎉 You did it! 🎉")
        .addClass('animate__animated animate__tada');
    
    $instructionText.html("You've learned how to cross the road safely!")
        .addClass('animate__animated animate__fadeIn');
    
    $poseToPerform.html(`Amazing job! You earned <span class="badge bg-warning text-dark">${gameState.points} points</span>!`);
    $poseToPerform.parent()
        .removeClass('alert-info')
        .addClass('alert-success animate__animated animate__pulse');
    
    // Play a congratulations sound if available
    if (successSound) {
        playSound(successSound);
    }
    
    // Speak congratulations message
    speak("Congratulations! You've successfully completed the exercise!")
        .then(() => {
            console.log("Completion message speech finished");
            
            // Level up if applicable - do this after speech
            if (gameState.totalCrossings % 3 === 0) {
                gameState.level++;
                $level.text(gameState.level)
                    .addClass('animate__animated animate__flip');
                
                setTimeout(() => {
                    $level.removeClass('animate__animated animate__flip');
                }, 1000);
                
                // Speak level up message if leveled up
                speak("Great job! You've leveled up!");
            }
            
            // Show reset button with animation after speech
            $resetButton.text("Try Again")
                .removeClass('d-none')
                .addClass('animate__animated animate__fadeIn')
                .off('click')
                .on('click', function() {
                    console.log("Reset button clicked");
                    // Remove any animation classes
                    $resetButton.removeClass('animate__animated animate__fadeIn');
                    // Make sure to set isActualReset to true to trigger proper reset
                    resetGame(true);
                });
            
            updateUIState();
        })
        .catch(error => {
            console.error("Error during completion speech:", error);
            
            // Still proceed with level up and button display even if speech fails
            if (gameState.totalCrossings % 3 === 0) {
                gameState.level++;
                $level.text(gameState.level)
                    .addClass('animate__animated animate__flip');
                
                setTimeout(() => {
                    $level.removeClass('animate__animated animate__flip');
                }, 1000);
            }
            
            $resetButton.text("Try Again")
                .removeClass('d-none')
                .addClass('animate__animated animate__fadeIn')
                .off('click')
                .on('click', function() {
                    console.log("Reset button clicked");
                    $resetButton.removeClass('animate__animated animate__fadeIn');
                    resetGame(true);
                });
            
            updateUIState();
        });
}

// Check the current pose
async function checkPose() {
    if (!gameState.waitingForPose || !gameState.started) return;
    
    try {
        // Record the current time for tracking successful connections
        const requestStartTime = Date.now();
        
        const response = await fetch('/get_pose', { 
            timeout: 5000,
            headers: { 'Cache-Control': 'no-cache' }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log("Pose data received:", data);
        
        // Reset connection attempts on success
        gameState.connectionAttempts = 0;
        gameState.isReconnecting = false;
        gameState.lastSuccessfulConnection = Date.now();
        
        // Remove any error notifications that might be showing
        removeConnectionErrorNotification();
        
        // Process the pose data (existing code)
        if (data && data.success) {
            const poseName = data.pose_name;
            const confidence = data.confidence;
            
            // Update UI to show current pose
            $currentPose.text(`${poseName} (${Math.round(confidence * 100)}%)`);
            
            // Store the current detected pose for character mirroring
            gameState.currentDetectedPose = poseName;
            
            // Update character based on detected pose
            updateCharacterDisplay();
            
            // Check if pose is correct
            const currentStep = config.gameSteps[gameState.currentStep];
            const requiredPose = currentStep.poseRequired;
            
            if (poseName === requiredPose && confidence >= config.poseConfidenceThreshold) {
                console.log(`Correct pose detected: ${poseName} with confidence ${confidence}`);
                handleCorrectPose();
            }
        }
        
        // Continue checking poses if game is still active
        if (gameState.waitingForPose && gameState.started) {
            setTimeout(checkPose, config.pollInterval);
        }
        
    } catch (error) {
        console.error("Error during pose check:", error);
        
        gameState.connectionAttempts++;
        const errorMessage = error.message || "Connection error";
        
        // Show error notification to user
        showConnectionErrorNotification(errorMessage);
        
        if (gameState.connectionAttempts <= gameState.maxConnectionAttempts) {
            // Try to reconnect with increasing delay
            const reconnectDelay = Math.min(
                gameState.reconnectDelay * Math.pow(1.5, gameState.connectionAttempts-1), 
                10000
            );
            
            console.log(`Reconnect attempt ${gameState.connectionAttempts} of ${gameState.maxConnectionAttempts} in ${reconnectDelay}ms`);
            gameState.isReconnecting = true;
            
            // Update UI to show reconnecting status
            $currentPose.text(`Reconnecting... (${gameState.connectionAttempts}/${gameState.maxConnectionAttempts})`);
            
            // Try again after delay
            setTimeout(checkPose, reconnectDelay);
        } else if (!gameState.isReconnecting) {
            // Max attempts reached, show failure and recover
            console.log("Max reconnection attempts reached, attempting recovery");
            gameState.isReconnecting = false;
            recoverFromConnectionFailure();
        }
    }
}

// Add these new functions for error handling and recovery
function showConnectionErrorNotification(message) {
    // Remove any existing notification first
    removeConnectionErrorNotification();
    
    // Create a new error notification
    const errorDiv = document.createElement('div');
    errorDiv.id = 'connectionErrorNotification';
    errorDiv.className = 'alert alert-warning connection-error';
    errorDiv.innerHTML = `
        <strong><i class="fas fa-exclamation-triangle"></i> Connection Issue</strong>
        <p>${message}</p>
        <p>Attempting to reconnect...</p>
        <div class="progress mt-2" style="height: 5px;">
            <div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" 
                 role="progressbar" style="width: 100%"></div>
        </div>
    `;
    
    // Add to the page
    document.body.appendChild(errorDiv);
}

function removeConnectionErrorNotification() {
    const existingNotification = document.getElementById('connectionErrorNotification');
    if (existingNotification) {
        existingNotification.remove();
    }
}

function recoverFromConnectionFailure() {
    // First, reset the connection attempts
    gameState.connectionAttempts = 0;
    
    // Show a more permanent error message with recovery options
    removeConnectionErrorNotification();
    
    const recoveryDiv = document.createElement('div');
    recoveryDiv.id = 'connectionRecoveryNotification';
    recoveryDiv.className = 'alert alert-danger connection-recovery';
    recoveryDiv.innerHTML = `
        <strong><i class="fas fa-exclamation-circle"></i> Connection Lost</strong>
        <p>We're having trouble connecting to the pose detection service.</p>
        <div class="d-flex justify-content-center mt-2">
            <button id="retryConnectionBtn" class="btn btn-warning me-2">
                <i class="fas fa-sync"></i> Retry Connection
            </button>
            <button id="resumeWithoutPoseBtn" class="btn btn-secondary">
                <i class="fas fa-forward"></i> Continue Anyway
            </button>
        </div>
    `;
    
    // Add to the page
    document.body.appendChild(recoveryDiv);
    
    // Add event listeners for recovery buttons
    document.getElementById('retryConnectionBtn').addEventListener('click', function() {
        // Hide the recovery notification
        recoveryDiv.remove();
        
        // Reset connection state and try again
        gameState.connectionAttempts = 0;
        gameState.isReconnecting = false;
        
        // Try to check pose again
        checkPose();
    });
    
    document.getElementById('resumeWithoutPoseBtn').addEventListener('click', function() {
        // Hide the recovery notification
        recoveryDiv.remove();
        
        // Move to the next step anyway (skip current pose check)
        if (gameState.waitingForPose && gameState.started) {
            speak("Let's continue to the next step.").then(() => {
                moveToNextStep();
            });
        } else {
            // If not in a waiting state, just try to reset everything
            resetGame();
            $startButton.removeClass('d-none');
        }
    });
}

// Add CSS for the error notifications
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.textContent = `
        .connection-error {
            position: fixed;
            top: 70px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            width: 350px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            animation: fadeIn 0.5s;
        }
        
        .connection-recovery {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 9999;
            width: 350px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            padding: 20px;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    `;
    document.head.appendChild(style);
});

// Modify the startPoseDetection function to reset connection state
function startPoseDetection() {
    if (!gameState.started) return;
    
    // Reset connection tracking
    gameState.connectionAttempts = 0;
    gameState.isReconnecting = false;
    gameState.lastSuccessfulConnection = Date.now();
    
    // Start checking pose
    checkPose();
}

// Update character pose
function updateCharacterPose(pose) {
    try {
        if (!pose) {
            console.error("Empty pose provided to updateCharacterPose");
            return;
        }
        
        console.log("Updating character pose to:", pose);
        
        // Map the pose name to a character class
        let characterPoseClass = '';
        let useArmElement = false;
        let armSide = '';
        
        switch (pose) {
            case 'Looking Left':
                characterPoseClass = 'look-left';
                console.log("Applying look-left pose");
                break;
                
            case 'Looking Right':
                characterPoseClass = 'look-right';
                console.log("Applying look-right pose");
                break;
                
            case 'Looking Down':
                characterPoseClass = 'look-down';
                console.log("Applying look-down pose");
                break;
                
            case 'Looking Straight':
                characterPoseClass = 'look-straight';
                console.log("Applying look-straight pose");
                break;
                
            case 'Raise Right Hand':
            case 'Hand Up':
                characterPoseClass = 'look-straight';
                useArmElement = true;
                armSide = 'right';
                console.log("Applying raise-right-hand pose");
                break;
                
            case 'Raise Left Hand':
                characterPoseClass = 'look-straight';
                useArmElement = true;
                armSide = 'left';
                console.log("Applying raise-left-hand pose");
                break;
                
            default:
                // Default to look-straight if unknown pose
                console.warn("Unknown pose:", pose, "defaulting to look-straight");
                characterPoseClass = 'look-straight';
                break;
        }
        
        // Remove all pose classes first
        $character.removeClass('look-left look-right look-down look-straight');
        
        // Remove any arm elements that might have been added
        $character.find('.right-arm-raised, .left-arm-raised').remove();
        
        // Apply the new pose class
        $character.addClass(characterPoseClass);
        
        // Add arm element if needed
        if (useArmElement) {
            if (armSide === 'right') {
                $character.append('<div class="right-arm-raised"></div>');
            } else if (armSide === 'left') {
                $character.append('<div class="left-arm-raised"></div>');
            }
        }
    } catch (error) {
        console.error("Error updating character pose:", error);
    }
}

// Move character
function moveCharacterForward(percentage) {
    try {
        const roadWidth = $("#roadAnimation").width();
        if (!roadWidth) {
            console.error("Road animation element not found or has no width");
            return;
        }
        
        const characterWidth = $character.width() || 30;
        const position = (roadWidth - characterWidth) * (percentage / 100);
        console.log(`Moving character to position ${position}px (${percentage}%)`);
        $character.css("left", position + "px");
    } catch (error) {
        console.error("Error moving character:", error);
    }
}

// Reset character
function resetCharacter() {
    try {
        $character.attr("class", "character").css("left", "10px");
    } catch (error) {
        console.error("Error resetting character:", error);
    }
}

// Start background music
function startBackgroundMusic() {
    try {
        if (isMusicEnabled) {
            backgroundMusic.play().catch(error => {
                console.error('Error playing background music:', error);
            });
        }
    } catch (error) {
        console.error('Error starting background music:', error);
    }
}

// Stop background music
function stopBackgroundMusic() {
    try {
        backgroundMusic.pause();
        backgroundMusic.currentTime = 0;
    } catch (error) {
        console.error('Error stopping background music:', error);
    }
}

// Start traffic sound
function startTrafficSound() {
    try {
        if (isSoundEnabled) {
            trafficSound.play().catch(error => {
                console.error('Error playing traffic sound:', error);
            });
        }
    } catch (error) {
        console.error('Error starting traffic sound:', error);
    }
}

// Stop traffic sound
function stopTrafficSound() {
    try {
        trafficSound.pause();
        trafficSound.currentTime = 0;
    } catch (error) {
        console.error('Error stopping traffic sound:', error);
    }
}

// Function to set up the initial state of the application
function initialSetup() {
    try {
        console.log("Initial setup starting...");
        // Reset any state to make sure we start fresh
        gameState.initialized = false;
        gameState.personDetected = false;
        gameState.lookingStraight = false;
        gameState.connectionAttempts = 0;
        
        // Disable the start button until the person is in position
        $startButton.prop('disabled', true)
            .removeClass('btn-primary animate__animated animate__pulse animate__infinite')
            .addClass('btn-secondary');
        
        // Set initial message
        $initialStatus.text("Please stand in front of the camera");
        $currentPose.text("Waiting for user...");
        
        // Ensure camera view has the awaiting-person class
        $("#cameraContainer").addClass("awaiting-person");
        
        // Start looking for a person - this is stage 1
        checkInitialPose();
        
    } catch (error) {
        console.error("Error in initialSetup:", error);
    }
}

// Enhance initGreeting to use sequential promises for speech
function initGreeting() {
    // Only run once
    if (gameState.initialized) {
        console.log("Greeting already initialized, skipping");
        return;
    }
    
    // Mark as initialized to prevent duplicate greetings
    gameState.initialized = true;
    gameState.personDetected = true;
    gameState.isGreetingInProgress = true; // Flag to prevent interruptions
    
    try {
        console.log("Initializing greeting...");
        
        // Ensure speech synthesis is available
        if (!window.speechSynthesis) {
            console.warn("Speech synthesis not available in this browser");
        } else {
            console.log("Speech synthesis available");
            
            // Force voice loading before attempting to speak
            loadVoices();
            
            // Force voice loading if needed
            if (availableVoices.length === 0) {
                console.log("No voices loaded yet, forcing voice load");
                speechSynthesis.getVoices();
                loadVoices();
            }
        }
        
        // Remove the gray overlay from camera
        $("#cameraContainer").removeClass("awaiting-person");
        
        // Set initial UI state
        updateUIState();
        
        // Initialize audio controls
        initAudioControls();
        
        // Start background music at reduced volume (30%)
        if (isMusicEnabled) {
            backgroundMusic.volume = 0.3;
            console.log("Setting background music volume to 30%");
            backgroundMusic.play().catch(err => console.log('Background music play failed:', err));
        }
        
        // Clear any existing speech queue and cancel any ongoing speech
        speechSynthesis.cancel();
        speechQueue = [];
        isSpeaking = false;
        
        // Clear any existing visual indicators
        $('.speech-indicator').remove();
        
        // Setup UI first before speech
        $initialStatus.text("I can see you!");
        
        // Pre-warm the speech synthesis with a silent utterance
        const warmUpUtterance = new SpeechSynthesisUtterance('');
        if (selectedVoice) {
            warmUpUtterance.voice = selectedVoice;
        }
        speechSynthesis.speak(warmUpUtterance);
        
        // Small delay to allow UI to update and voice engine to initialize
        setTimeout(() => {
            // Reset speech flags
            gameState.greetingSpeechComplete = false;
            gameState.lookStraightSpeechStarted = false;
            gameState.lookStraightSpeechComplete = false;
            
            // First greeting with sequential promises
            console.log("Starting greeting speech sequence");
            
            // First greeting - just acknowledge we can see the person
            speak("Hello there! I can see you!")
                .then(() => {
                    console.log("Initial greeting complete, waiting before next speech");
                    
                    // Update UI for the next instruction
                    $initialStatus.text("Now, please look straight ahead");
                    
                    // Return a promise that resolves after a short delay
                    return new Promise(resolve => setTimeout(resolve, 800));
                })
                .then(() => {
                    // Now we can start the next speech segment
                    console.log("Starting looking straight instruction");
                    gameState.lookStraightSpeechStarted = true;
                    
                    // Return the promise from the next speak call
                    return speak("Now, please look straight ahead so we can begin our road safety exercise.");
                })
                .then(() => {
                    console.log("Looking straight instruction complete");
                    
                    // Both speech segments are complete
                    gameState.greetingSpeechComplete = true;
                    gameState.lookStraightSpeechComplete = true;
                    gameState.isGreetingInProgress = false;
                    
                    // Update UI after both speech sections complete
                    $poseToPerform.text('Look straight ahead to continue');
                    $poseToPerform.parent()
                        .removeClass('alert-success')
                        .addClass('alert-warning');
                    updateCharacterPose('look-straight');
                })
                .catch(error => {
                    console.error("Error during greeting speech sequence:", error);
                    
                    // Mark all speech as complete to allow flow to continue even if there's an error
                    gameState.greetingSpeechComplete = true;
                    gameState.lookStraightSpeechComplete = true;
                    gameState.isGreetingInProgress = false;
                    
                    // Still update UI even if speech fails
                    $initialStatus.text("Please look straight ahead to continue");
                    $poseToPerform.text('Look straight ahead to continue');
                    $poseToPerform.parent().addClass('alert-warning');
                    updateCharacterPose('look-straight');
                });
        }, 1000); // Delay to ensure UI updates and voices load before speaking
        
        console.log("Greeting initialization complete");
    } catch (error) {
        console.error("Error during greeting initialization:", error);
        
        // Attempt to recover gracefully
        gameState.greetingSpeechComplete = true;
        gameState.lookStraightSpeechComplete = true;
        gameState.isGreetingInProgress = false;
        alert("There was an error initializing the greeting. Please refresh the page and try again.");
    }
}

// Also update the startBackgroundMusic function to respect the volume setting
function startBackgroundMusic() {
    try {
        if (isMusicEnabled) {
            // Set volume to 30% if a person is detected
            if (gameState.personDetected) {
                backgroundMusic.volume = 0.3;
                console.log("Starting background music at 30% volume");
            } else {
                backgroundMusic.volume = 1.0;
                console.log("Starting background music at 100% volume");
            }
            
            backgroundMusic.play().catch(error => {
                console.error('Error playing background music:', error);
            });
        }
    } catch (error) {
        console.error('Error starting background music:', error);
    }
}

// Update the checkInitialPose function to reduce music volume when a person is detected
async function checkInitialPose() {
    console.log("Checking for initial pose...", new Date().toISOString());
    
    // Don't run checks if game already started or greeting speech is in progress
    if (gameState.started || gameState.isGreetingInProgress) {
        console.log("Game started or greeting in progress, skipping pose check");
        
        // If greeting is in progress, check back soon
        if (gameState.isGreetingInProgress) {
            setTimeout(checkInitialPose, INIT_CHECK_INTERVAL);
        }
        
        return;
    }

    try {
        console.log("Fetching pose data...");
        
        // Add timeout to the fetch request
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
        
        const response = await fetch('/get_pose', { 
            signal: controller.signal,
            headers: { 'Cache-Control': 'no-cache' }
        });
        
        clearTimeout(timeoutId); // Clear the timeout
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        console.log("Initial pose check response:", data);

        // Reset error state since we got a valid response
        gameState.connectionAttempts = 0;
        removeConnectionErrorNotification();

        // First stage: Just detect any person (Stage 1 - Detection)
        if (!gameState.initialized) {
            console.log("First stage: Checking for person presence");
            // Check if any pose is detected with reasonable confidence (person is in frame)
            if (data && data.success && data.pose_name && data.confidence >= 0.3) {
                console.log(`Person detected with pose: ${data.pose_name} and confidence ${data.confidence}`);
                
                // Set person detected flag
                gameState.personDetected = true;
                
                // Lower background music volume if it's playing
                if (backgroundMusic && !backgroundMusic.paused) {
                    backgroundMusic.volume = 0.2;
                    console.log("Person detected: Reducing background music to 30% volume");
                }
                
                // Initialize greeting and move to stage 2 - this triggers the speech
                initGreeting();
                
                // Continue to the next phase to check if greeting is complete
                setTimeout(checkInitialPose, INIT_CHECK_INTERVAL);
                return;
            } else {
                // Keep checking for a person
                console.log("No person detected yet or confidence too low");
                $initialStatus.text("Please stand where I can see you");
                
                // Gray out camera view until person is detected
                $("#cameraContainer").addClass("awaiting-person");
                
                if (data && data.pose_name) {
                    $currentPose.text(`${data.pose_name} (${Math.round(data.confidence * 100)}%)`);
                } else {
                    $currentPose.text('Waiting for user...');
                }
                
                // Continue checking
                setTimeout(checkInitialPose, INIT_CHECK_INTERVAL);
                return;
            }
        }
        // Make sure greeting speech is complete before proceeding
        else if (!gameState.greetingSpeechComplete || !gameState.lookStraightSpeechComplete) {
            console.log("Waiting for greeting speech to complete");
            
            // Just update pose display but don't proceed to next stage
            if (data && data.success && data.pose_name) {
                $currentPose.text(`${data.pose_name} (${Math.round(data.confidence * 100)}%)`);
            }
            
            // Check again after a delay
            setTimeout(checkInitialPose, INIT_CHECK_INTERVAL);
            return;
        }
        // Second stage: Require "Looking Straight" pose to enable start button (Stage 2 - Position)
        else {
            console.log("Second stage: Checking for Looking Straight pose");
            // Check if we have valid data with "Looking Straight" pose
            if (data && data.success && data.pose_name === INITIAL_POSE && data.confidence >= config.poseConfidenceThreshold) {
                console.log(`Initial pose "${INITIAL_POSE}" detected with confidence ${data.confidence}`);
                
                // Update UI to show success
                $initialStatus.text("Perfect! You're looking straight ahead.");
                $currentPose.text(`${data.pose_name} (${Math.round(data.confidence * 100)}%)`);
                
                // Show success indicator 
                $successFeedback.css('opacity', 1);
                setTimeout(() => {
                    $successFeedback.css('opacity', 0);
                }, 1000);
                
                // Update character to match detected pose
                updateCharacterPose(data.pose_name);
                
                // Check if the start button is already enabled
                if ($startButton.prop('disabled') && !gameState.startButtonSpeechInProgress && !gameState.startButtonSpeechComplete) {
                    // First time detecting the correct pose
                    // Set the flag to prevent interruptions
                    gameState.startButtonSpeechInProgress = true;
                    gameState.startButtonSpeechComplete = false;
                    
                    // Speak instructions to press start button
                    speak("Perfect! You're looking straight ahead. Now press the Start Exercise button!")
                        .then(() => {
                            console.log("Start button instruction speech complete");
                            // Update flags after speech completes
                            gameState.startButtonSpeechComplete = true;
                            gameState.startButtonSpeechInProgress = false;
                            // Enable start button after speech
                            enableStartButton();
                        })
                        .catch(error => {
                            console.error("Error during start button speech:", error);
                            // Still enable the button even if speech fails
                            gameState.startButtonSpeechComplete = true;
                            gameState.startButtonSpeechInProgress = false;
                            enableStartButton();
                        });
                }
                
                // Keep checking in case they look away before pressing start
                setTimeout(checkInitialPose, INIT_CHECK_INTERVAL);
                return;
            } 
            else if (data && data.success && data.pose_name) {
                // Detected a pose but not looking straight
                console.log(`Detected ${data.pose_name} but need Looking Straight. Confidence: ${data.confidence}`);
                $initialStatus.text(`Please look straight ahead`);
                $currentPose.text(`${data.pose_name} (${Math.round(data.confidence * 100)}%)`);
                
                // Update instruction with clear guidance
                $poseToPerform.text('Look straight ahead to continue');
                $poseToPerform.parent()
                    .removeClass('alert-success')
                    .addClass('alert-warning');
                
                // Update character to show what's needed
                updateCharacterPose('look-straight');
                
                // Disable start button
                $startButton.prop('disabled', true)
                    .removeClass('btn-primary animate__animated animate__pulse animate__infinite')
                    .addClass('btn-secondary');
                
                // Continue checking
                setTimeout(checkInitialPose, INIT_CHECK_INTERVAL);
                return;
            }
            else {
                // Keep checking for a person
                console.log("Lost track of person or invalid data");
                $initialStatus.text("Please stand where I can see you clearly");
                
                // Add gray overlay since we lost the person or clear view
                $("#cameraContainer").addClass("awaiting-person");
                
                // Continue checking
                setTimeout(checkInitialPose, INIT_CHECK_INTERVAL);
                return;
            }
        }

    } catch (error) {
        console.error("Error during initial pose check:", error);
        
        // Track connection attempt failures
        gameState.connectionAttempts++;
        const maxAttempts = 5; // Maximum number of consecutive attempts before showing recovery UI
        
        // Don't keep retrying forever; show a recovery UI after several attempts
        if (gameState.connectionAttempts >= maxAttempts) {
            console.warn(`Connection failed ${gameState.connectionAttempts} times, showing recovery UI`);
            
            // Show more permanent error with recovery options
            showInitializationErrorUI();
            
            // Don't schedule another check automatically; let user retry manually
            return;
        }
        
        // Show user-friendly message based on error type
        let errorMessage = "Connection issue. Retrying...";
        if (error.name === 'AbortError') {
            errorMessage = "Connection timed out. Retrying...";
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = "Server unavailable. Retrying...";
        }
        
        $initialStatus.text(errorMessage);
        $currentPose.text("Connection error");
        
        // Show error notification
        showConnectionErrorNotification(errorMessage);
        
        // Show the awaiting-person overlay due to error
        $("#cameraContainer").addClass("awaiting-person")
            .addClass("connection-error");
        
        // Exponential backoff for retries
        const retryDelay = Math.min(INIT_CHECK_INTERVAL * Math.pow(1.5, gameState.connectionAttempts), 5000);
        console.log(`Will retry in ${retryDelay}ms (attempt ${gameState.connectionAttempts})`);
        
        // Continue checking with increasing delay between retries
        if (!gameState.started) {
            setTimeout(checkInitialPose, retryDelay);
        }
    }
}

// Helper function to enable start button with animation
function enableStartButton() {
    // Enable start button
    $startButton.prop('disabled', false)
        .removeClass('btn-secondary')
        .addClass('btn-primary animate__animated animate__pulse animate__infinite');
    
    // Show clear instruction that user should press start
    $poseToPerform.text('Press the Start Exercise button!');
    $poseToPerform.parent()
        .removeClass('alert-warning')
        .addClass('alert-success animate__animated animate__pulse');
        
    // Remove gray overlay
    $("#cameraContainer").removeClass("awaiting-person");
}

// New function to show initialization error UI with recovery options
function showInitializationErrorUI() {
    // Remove any existing notifications
    removeConnectionErrorNotification();
    
    // Create an error card for initialization failures
    const errorCard = document.createElement('div');
    errorCard.id = 'initializationErrorCard';
    errorCard.className = 'card error-card animate__animated animate__fadeIn';
    errorCard.innerHTML = `
        <div class="card-header bg-danger text-white">
            <h5 class="m-0"><i class="fas fa-exclamation-triangle"></i> Connection Problem</h5>
        </div>
        <div class="card-body">
            <p>We're having trouble connecting to the pose detection service.</p>
            <p>This could be due to:</p>
            <ul>
                <li>Server not running</li>
                <li>Network connection issues</li>
                <li>Browser permissions not granted</li>
            </ul>
            <div class="d-flex justify-content-between mt-3">
                <button id="manualStartBtn" class="btn btn-warning">
                    <i class="fas fa-play-circle"></i> Start Without Pose Detection
                </button>
                <button id="retryConnectionBtn" class="btn btn-primary">
                    <i class="fas fa-sync"></i> Try Again
                </button>
            </div>
        </div>
    `;
    
    // Add the error card to the page
    document.body.appendChild(errorCard);
    
    // Add event listeners to buttons
    document.getElementById('retryConnectionBtn').addEventListener('click', function() {
        // Remove the error card
        errorCard.remove();
        
        // Reset connection attempts
        gameState.connectionAttempts = 0;
        
        // Try again
        checkInitialPose();
    });
    
    document.getElementById('manualStartBtn').addEventListener('click', function() {
        // Remove the error card
        errorCard.remove();
        
        // Enable start button and let user proceed without pose detection
        $startButton.prop('disabled', false)
            .removeClass('btn-secondary')
            .addClass('btn-primary animate__animated animate__pulse animate__infinite');
        
        // Update UI to indicate manual mode
        $initialStatus.text("Pose detection unavailable. You can start manually.");
        $currentPose.text("Manual mode");
        
        // Remove error styling
        $("#cameraContainer").removeClass("connection-error");
        
        // Add warning that pose detection isn't working
        const warningBanner = document.createElement('div');
        warningBanner.className = 'alert alert-warning mt-2';
        warningBanner.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Pose detection unavailable. Some features may not work.';
        document.querySelector('#cameraContainer').appendChild(warningBanner);
    });
    
    // Add CSS for the error card
    if (!document.getElementById('errorCardStyles')) {
        const style = document.createElement('style');
        style.id = 'errorCardStyles';
        style.textContent = `
            .error-card {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 9999;
                width: 400px;
                max-width: 90vw;
                box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            }
            
            .connection-error {
                border: 2px solid #ff5c5c;
                box-shadow: 0 0 10px rgba(255, 92, 92, 0.5);
            }
        `;
        document.head.appendChild(style);
    }
}

// Update UI state based on game state
function updateUIState() {
    try {
        console.log("Updating UI state, game started:", gameState.started);
        
        // Update game controls based on game state
        if (gameState.started) {
            $startButton.addClass('d-none');
            $resetButton.addClass('d-none');
        } else {
            $startButton.removeClass('d-none');
            if (gameState.totalCrossings > 0) {
                $resetButton.removeClass('d-none');
            } else {
                $resetButton.addClass('d-none');
            }
        }
        
        // Update score display with max points
        $points.text(`${gameState.points}/${MAX_POINTS}`);
        $level.text(gameState.level);
        
        // Update progress bar
        updateProgressBar();
        
        console.log("UI state updated successfully");
    } catch (error) {
        console.error("Error updating UI state:", error);
    }
}

// Initialize character
function renderCharacter() {
    try {
        console.log("Rendering character");
        
        // Reset character position and state
        resetCharacter();
        
        // Make sure character is visible
        $character.removeClass('d-none');
        
        // Set initial character position
        $character.css({
            left: '10px',
            bottom: '10px'
        });
        
        console.log("Character rendering complete");
    } catch (error) {
        console.error("Error rendering character:", error);
    }
}

// Update progress bar based on current step
function updateProgressBar() {
    try {
        if (!config.gameSteps.length) return;
        
        const progressPercentage = (gameState.currentStep / config.gameSteps.length) * 100;
        $progressBar.css('width', `${progressPercentage}%`);
        
        // Update progress bar color based on progress
        if (progressPercentage >= 75) {
            $progressBar.removeClass('bg-danger bg-warning').addClass('bg-success');
        } else if (progressPercentage >= 40) {
            $progressBar.removeClass('bg-danger bg-success').addClass('bg-warning');
        } else {
            $progressBar.removeClass('bg-warning bg-success').addClass('bg-danger');
        }
    } catch (error) {
        console.error("Error updating progress bar:", error);
    }
}

// Update character display based on student's detected pose
function updateCharacterDisplay() {
    try {
        // Only update if a pose is detected
        if (!gameState.currentDetectedPose) {
            return;
        }
        
        console.log("Updating character display based on pose:", gameState.currentDetectedPose);
        
        // Map detected pose to character pose class
        let characterPoseClass = '';
        let useArmElement = false;
        let armSide = '';
        
        switch (gameState.currentDetectedPose) {
            case 'Looking Left':
                characterPoseClass = 'look-left';
                break;
                
            case 'Looking Right':
                characterPoseClass = 'look-right';
                break;
                
            case 'Looking Down':
                characterPoseClass = 'look-down';
                break;
                
            case 'Looking Straight':
                characterPoseClass = 'look-straight';
                break;
                
            case 'Raise Right Hand':
            case 'Hand Up':
                characterPoseClass = 'look-straight';
                useArmElement = true;
                armSide = 'right';
                break;
                
            case 'Raise Left Hand':
                characterPoseClass = 'look-straight';
                useArmElement = true;
                armSide = 'left';
                break;
                
            case 'Walking':
                // For walking pose, animate the character moving forward
                const currentPosition = parseInt($character.css('left')) || 10;
                const newPosition = currentPosition + 5;
                $character.css('left', newPosition + 'px');
                break;
                
            default:
                // No matching pose, don't update character
                return;
        }
        
        // If in a game step, don't override the required pose demonstration
        if (gameState.waitingForPose && gameState.started) {
            const currentStep = config.gameSteps[gameState.currentStep];
            
            // Check if the detected pose matches the required pose
            if (gameState.currentDetectedPose === currentStep.poseRequired) {
                // If it matches, enhance the character to show "good job" feedback
                $character.addClass('character-correct');
                setTimeout(() => {
                    $character.removeClass('character-correct');
                }, 500);
            }
            
            // Don't override the character pose during step demonstration
            return;
        }
        
        // Outside of specific game steps, mirror the student's pose
        if (characterPoseClass) {
            // Remove all pose classes first
            $character.removeClass('look-left look-right look-down look-straight');
            
            // Remove any arm elements that might have been added
            $character.find('.right-arm-raised, .left-arm-raised').remove();
            
            // Apply the new pose class
            $character.addClass(characterPoseClass);
            
            // Add arm element if needed
            if (useArmElement) {
                if (armSide === 'right') {
                    $character.append('<div class="right-arm-raised"></div>');
                } else if (armSide === 'left') {
                    $character.append('<div class="left-arm-raised"></div>');
                }
            }
        }
    } catch (error) {
        console.error("Error updating character display:", error);
    }
}

// Add CSS for point deduction animation
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.textContent += `
        .point-deduction {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 32px;
            font-weight: bold;
            color: #ff3b30;
            text-shadow: 0 0 5px rgba(0,0,0,0.5);
            z-index: 1000;
            opacity: 1;
        }
        
        .character-timeout {
            filter: grayscale(50%);
            transform: scale(0.95);
            transition: all 0.3s ease;
        }
        
        .step-timer {
            transition: width 0.1s linear, background-color 0.5s ease;
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
        
        .timeout-alert {
            animation: shake 0.5s;
            background-color: rgba(255, 59, 48, 0.8) !important;
            border-color: #ff3b30 !important;
        }
    `;
    document.head.appendChild(style);
});

// Add constant for maximum possible points
const MAX_POINTS = config.gameSteps.reduce((total, step) => total + step.points, 0);

// Add this new section near the top after document ready
// Camera handling code for frame submission
let videoElement;
let canvasElement;
let canvasContext;
let isCapturingFrames = false;
let frameSubmissionInterval = 200; // milliseconds between frame submissions
let frameIntervalId = null;

// Setup camera capture once document is ready
function setupCameraCapture() {
    console.log('Setting up camera capture for frame submission');
    
    // Create hidden video and canvas elements for frame capture
    videoElement = document.createElement('video');
    videoElement.id = 'cameraVideo';
    videoElement.style.display = 'none';
    videoElement.autoplay = true;
    videoElement.playsinline = true;
    document.body.appendChild(videoElement);
    
    canvasElement = document.createElement('canvas');
    canvasElement.id = 'frameCanvas';
    canvasElement.style.display = 'none';
    canvasElement.width = 640;
    canvasElement.height = 480;
    document.body.appendChild(canvasElement);
    
    canvasContext = canvasElement.getContext('2d');
    
    // Request camera access
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(function(stream) {
                videoElement.srcObject = stream;
                console.log('Camera access granted');
                startFrameCapture();
            })
            .catch(function(error) {
                console.error('Error accessing camera:', error);
                showCameraError(error.message);
            });
    } else {
        console.error('getUserMedia not supported in this browser');
        showCameraError('Camera access not supported in this browser');
    }
}

// Start capturing and sending frames
function startFrameCapture() {
    if (isCapturingFrames) return;
    
    console.log('Starting frame capture');
    isCapturingFrames = true;
    
    // Capture and send frames at regular intervals
    frameIntervalId = setInterval(captureAndSendFrame, frameSubmissionInterval);
}

// Stop capturing frames
function stopFrameCapture() {
    if (!isCapturingFrames) return;
    
    console.log('Stopping frame capture');
    isCapturingFrames = false;
    
    if (frameIntervalId) {
        clearInterval(frameIntervalId);
        frameIntervalId = null;
    }
}

// Capture a frame and send it to the server
function captureAndSendFrame() {
    if (!videoElement || !canvasElement || !canvasContext || !isCapturingFrames) return;
    
    try {
        // Draw the current video frame to the canvas
        canvasContext.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        
        // Convert the canvas to a data URL (JPEG format)
        const frameDataUrl = canvasElement.toDataURL('image/jpeg', 0.8);
        
        // Send the frame to the server
        fetch('/submit_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                frame: frameDataUrl
            })
        })
        .then(response => response.json())
        .then(data => {
            // Handle the response if needed
            if (data.success && data.pose_name) {
                // Update UI with the detected pose
                gameState.connectionAttempts = 0;
                gameState.isReconnecting = false;
                gameState.lastSuccessfulConnection = Date.now();
                
                // Remove any error notifications
                removeConnectionErrorNotification();
                
                // Update the current pose display
                $currentPose.text(`${data.pose_name} (${Math.round(data.confidence * 100)}%)`);
                
                // Store the current detected pose for character mirroring
                gameState.currentDetectedPose = data.pose_name;
                
                // Update character based on detected pose
                updateCharacterDisplay();
                
                // If we're waiting for a pose in the game, check if it matches
                if (gameState.waitingForPose && gameState.started) {
                    // Check if pose is correct
                    const currentStep = config.gameSteps[gameState.currentStep];
                    const requiredPose = currentStep.poseRequired;
                    
                    if (data.pose_name === requiredPose && data.confidence >= config.poseConfidenceThreshold) {
                        console.log(`Correct pose detected: ${data.pose_name} with confidence ${data.confidence}`);
                        handleCorrectPose();
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error sending frame:', error);
            
            if (isCapturingFrames) {
                gameState.connectionAttempts++;
                
                if (gameState.connectionAttempts > gameState.maxConnectionAttempts) {
                    // Show error UI and attempt recovery
                    showConnectionErrorNotification('Error connecting to server: ' + error.message);
                }
            }
        });
    } catch (error) {
        console.error('Error capturing frame:', error);
    }
}

// Show camera error message
function showCameraError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.id = 'cameraErrorNotification';
    errorDiv.className = 'alert alert-danger camera-error';
    errorDiv.innerHTML = `
        <strong><i class="fas fa-video-slash"></i> Camera Error</strong>
        <p>${message}</p>
        <p>Please make sure your camera is connected and that you've granted permission to use it.</p>
        <button id="retryCameraBtn" class="btn btn-warning mt-2">
            <i class="fas fa-sync"></i> Retry Camera Access
        </button>
    `;
    
    // Add to the page
    document.body.appendChild(errorDiv);
    
    // Add event listener for retry button
    document.getElementById('retryCameraBtn').addEventListener('click', function() {
        // Remove the error notification
        errorDiv.remove();
        
        // Try setting up the camera again
        setupCameraCapture();
    });
}

// Modify the existing checkPose function to use our new direct frame submission
async function checkPose() {
    if (!gameState.waitingForPose || !gameState.started) return;
    
    // Our frame submission is now handled by captureAndSendFrame
    // This function is kept for compatibility but doesn't need to do the fetch anymore
    
    // Continue checking poses if game is still active
    if (gameState.waitingForPose && gameState.started) {
        setTimeout(checkPose, config.pollInterval);
    }
}

// Modify the initialSetup function to call setupCameraCapture
function initialSetup() {
    console.log("Running initial setup...");
    
    // Setup camera capture first
    setupCameraCapture();
    
    // Then run the initial checks for pose detection
    setTimeout(checkInitialPose, 1000);
}

