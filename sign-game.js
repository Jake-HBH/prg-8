import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const enableWebcamButton = document.getElementById("webcamButton")
const logButton = document.getElementById("logButton")

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")

const drawUtils = new DrawingUtils(canvasCtx)
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let nn
let totalPredictions = 0;
let correctPredictions = 0;
let questionIndex = 0;
// const correctLabel = "wave";

const questions = [
    {
        question: "What is the hand pose for 'Thumbs Up'?",
        correctAnswer: "thumbs up"
    },
    {
        question: "What is the hand pose for 'Peace'?",
        correctAnswer: "peace"
    },
    {
        question: "What is the hand pose for 'Fist'?",
        correctAnswer: "fist"
    },
    {
        question: "What is the hand pose for 'Thumbs Down'?",
        correctAnswer: "thumbs down"
    },
    {
        question: "What is the hand pose for 'Ok'?",
        correctAnswer: "ok"
    },
    {
        question: "What is the hand pose for 'Wave'?",
        correctAnswer: "wave"
    }
];

// vraag laten zien op scherm
function showQuestion() {
    const questionEl = document.getElementById("questionOutput");
    const question = questions[questionIndex];
    questionEl.textContent = question.question;
}

function showFeedback(isCorrect) {
    const feedbackEl = document.getElementById("feedbackOutput");
    if (isCorrect) {
        feedbackEl.textContent = "Correct! Great job!";
    } else {
        feedbackEl.textContent = "Oops! That's not right. Try again!";
    }

    // na feedback naar volgende vraag
    setTimeout(() => {
        questionIndex++;
        if (questionIndex >= questions.length) {
            questionIndex = 0; // door de vragen loopen, starten bij begin als het klaar is
        }
        showQuestion();
    }, 2000); // feedback laten zien voor 2 sec
}

function createNeuralNetwork(){
    ml5.setBackend("webgl")
    nn = ml5.neuralNetwork({task: 'classification', debug: true});
    const options = {
        model: "./model/model.json",
        metadata : "./model/model_meta.json",
        weights:"./model/model.weights.bin"
    }
    nn.load(options, createHandLandmarker())
}

let image = document.querySelector("#myimage")


/********************************************************************
 // CREATE THE POSE DETECTOR
 ********************************************************************/
const createHandLandmarker = async () => {
    console.log("neural network model is loaded")

    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
    console.log("model loaded, you can start webcam")

    enableWebcamButton.addEventListener("click", (e) => enableCam(e))
    logButton.addEventListener("click", (e) => classifyHand(e))
}

/********************************************************************
 // START THE WEBCAM
 ********************************************************************/
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

/********************************************************************
 // START PREDICTIONS
 ********************************************************************/
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now())

    let hand = results.landmarks[0]
    if(hand) {
        let thumb = hand[4]
        image.style.transform = `translate(${video.videoWidth - thumb.x * video.videoWidth}px, ${thumb.y * video.videoHeight}px)`
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    for(let hand of results.landmarks){
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam)
    }
}

/********************************************************************
 // LOG HAND COORDINATES IN THE CONSOLE
 ********************************************************************/
function logAllHands(){
    for (let hand of results.landmarks) {
        // console.log(hand)
        console.log(hand[4])
    }
}

function classifyHand() {
    console.log(results.landmarks[0]);
    let numbersOnly = [];
    let hand = results.landmarks[0];
    for (let point of hand) {
        numbersOnly.push(point.x, point.y, point.z);
    }

    nn.classify(numbersOnly, (results) => {
        const predictionDiv = document.getElementById("predictionOutput");
        const label = results[0].label;
        const confidence = (results[0].confidence * 100).toFixed(2);
        predictionDiv.textContent = `I think this pose is a "${label}" with ${confidence}% confidence.`;

        // een check als de verwachtte label matcht met het huidige correcte antwoord van de vraag die nu word getoond
        const correctAnswer = questions[questionIndex].correctAnswer;
        const isCorrect = label.toLowerCase() === correctAnswer.toLowerCase();

        // laat feedback zien van goed of fout
        showFeedback(isCorrect);

        // bereken en laat accuracy zien
        totalPredictions++;
        if (isCorrect) {
            correctPredictions++;
        }

        const accuracyEl = document.getElementById("accuracyOutput");
        const accuracy = (correctPredictions / totalPredictions) * 100;
        accuracyEl.textContent = `Accuracy so far: ${accuracy.toFixed(2)}%`;
    });
}

/********************************************************************
 // START THE APP
 ********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    createNeuralNetwork()
    showQuestion();
}