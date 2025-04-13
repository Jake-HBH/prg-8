import posedata from "./data.json" with {type: 'json'}
import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import kNear from "./knear.js"

const enableWebcamButton = document.getElementById("webcamButton");
const logButton = document.getElementById("logButton");
const predictButton = document.getElementById("predictButton");

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const drawUtils = new DrawingUtils(canvasCtx);
let handLandmarker;
let webcamRunning = false;
let handResults;

const k = 3
const ai = new kNear(k);

// leert de ai alle voorbeelden uit posedata
posedata.forEach(posedata => {
    ai.learn(posedata[0], posedata[1])
});

// laadt de hand herkenner
const initializeTrackers = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1 // kijken naar 1 hand tegelijk
    });

    // zodra alles klaar is, kan de webcam starten
    console.log("Modellen geladen, start nu de webcam");
    enableWebcamButton.addEventListener("click", toggleWebcam);
    logButton.addEventListener("click", logResults);
    predictButton.addEventListener("click", () => {
        if (handResults && handResults.landmarks) {
            let flatArray = handResults.landmarks[0].flatMap(p => [p.x, p.y, p.z]);
            calculateSign(flatArray, posedata);
        } else {
            console.warn("Geen hand gevonden om te voorspellen");
        }
    });
};

// webcam aan
async function toggleWebcam() {
    if (webcamRunning) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
        return;
    }

    webcamRunning = true;
    enableWebcamButton.innerText = "DISABLE PREDICTIONS";

    try {
        // vraag toestemming voor de webcam
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;

        // zodra video geladen is, starten we met voorspellen
        video.addEventListener("loadeddata", () => {
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            predictWebcam();
        });
    } catch (error) {
        console.error("fout bij openen van webcam:", error);
    }
}

// voorspelt telkens opnieuw zolang de webcam aan staat
async function predictWebcam() {
    if (!handLandmarker) return;

    handResults = await handLandmarker.detectForVideo(video, performance.now());

    drawResults(); // teken de hand
    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

// tekent de hand op het canvas
function drawResults() {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (handResults?.landmarks) {
        for (let hand of handResults.landmarks) {
            drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, {color: "#00FF00", lineWidth: 5});
            drawUtils.drawLandmarks(hand, {radius: 4, color: "#FF0000", lineWidth: 2});
        }
    }
}

// laat in de console zien wat de handpositie is
function logResults() {
    console.log("handpunten:", handResults?.landmarks);
    logToData()
}

// als je toestemming hebt voor de webcam, start dan de app
if (navigator.mediaDevices?.getUserMedia) {
    initializeTrackers();
}

// sla data op in localStorage met een label erbij
function logToData() {
    if (!handResults || !handResults.landmarks) {
        console.warn("geen handdata om op te slaan");
        return;
    }

    let newSignArray = [];
    handResults.landmarks[0].forEach(point => {
        newSignArray.push(point.x, point.y, point.z);
    });

    // haalt het label uit een invoerveld
    const labelInput = document.getElementById("labelInput");
    const label = labelInput.value.trim();

    // check of er een label is ingevuld
    if (!label) {
        alert("Typ eerst een label voordat je iets opslaat.");
        return;
    }

    // haalt oude data op (of maakt lege lijst)
    let allData = JSON.parse(localStorage.getItem("data")) || [];

    // voegt nieuwe data toe
    allData.push({
        data: newSignArray,
        label: label
    });

    // slaat alle data weer op in localStorage
    localStorage.setItem("data", JSON.stringify(allData));

    console.log(`opgeslagen met label "${label}"`);
}

