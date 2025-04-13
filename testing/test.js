import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// knoppen van de pagina
const trainButton = document.getElementById("trainButton")
const testButton = document.getElementById("testButton")

// alle handdata
let data = []           // alle data
let trainData = []      // data om te trainen
let testData = []       // data om te testen

// hoe het AI-model werkt (weggl)
ml5.setBackend("webgl")

const nn = ml5.neuralNetwork({ task: 'classification', debug: true })

// haal de data op uit data.json
function initData(){
    fetch("data.json") // open het bestand met poses
        .then(response => response.json())
        .then(d => {
            data = d // stop alles in de lijst
            trainButton.addEventListener("click", (e) => train())
        })
        .catch(error => console.log(error)) // als er iets fout gaat
}

// start trainen van de ai
function train(){
    // schud de data door elkaar met random
    data.sort(() => (Math.random() - 0.5))

    // 80% van de data om mee te trainen
    trainData = data.slice(0, Math.floor(data.length * 0.8))

    // de rest om te testen
    testData = data.slice(Math.floor(data.length * 0.8) + 1)

    // de trainData in het AI-model stoppen
    for(const {data, label} of trainData){
        nn.addData(data, {label: label})
        console.log(data, label)
    }

    nn.normalizeData()

    // laat het model leren/trainen
    nn.train({ epochs: 35 }, () => finishedTraining())
}

// test knop of het model goed werkt, accuracy laten zien
testButton.addEventListener("click", (e) => test())

// test het model met de testData
async function test(){
    let labels = [...new Set(testData.map(d => d.label))]; // haal alle soorten poses op
    let matrix = {}; // opslaan wat goed/fout ging
    let correct = 0; // tel hoeveel goed is

    // tabel
    for (let actual of labels) {
        matrix[actual] = {};
        for (let predicted of labels) {
            matrix[actual][predicted] = 0;
        }
    }

    // loop door de testdata
    for(const {data, label: actualLabel} of testData){
        const prediction = await nn.classify(data); // laat het model voorspellen
        const predictedLabel = prediction[0].label;

        matrix[actualLabel][predictedLabel]++; // tel wat het model zei

        if (actualLabel === predictedLabel) correct++; // goed voorspeld? tel op
    }

    // bereken hoe goed het model is
    const accuracy = ((correct / testData.length) * 100).toFixed(2);

    // laat de resultaten zien
    renderConfusionMatrix(matrix, labels, accuracy);
}

// laat de confusion matrix zien met de resultaten
function renderConfusionMatrix(matrix, labels, accuracy){
    const table = document.getElementById("confusionMatrix");
    table.innerHTML = ""; // maak leeg

    // titel rij
    let header = "<tr><th>Goed \\ Gegokt</th>";
    for(let label of labels){
        header += `<th>${label}</th>`;
    }
    header += "</tr>";
    table.innerHTML += header;

    // alle rijen met cijfers
    for(let actual of labels){
        let row = `<tr><th>${actual}</th>`;
        for(let predicted of labels){
            row += `<td>${matrix[actual][predicted]}</td>`;
        }
        row += "</tr>";
        table.innerHTML += row;
    }

    // laat de score zien
    const accuracyDisplay = document.createElement("p");
    accuracyDisplay.id = "accuracyDisplay";
    accuracyDisplay.innerText = `Goed geraden: ${accuracy}%`;
    document.getElementById("confusionMatrixContainer").appendChild(accuracyDisplay);
}

// als trainen klaar is
function finishedTraining() {
    console.log("AI is klaar met leren!");
    nn.save("model", () => console.log("model is opgeslagen"))
}

// start alles als je camera toegang hebt
if (navigator.mediaDevices?.getUserMedia) {
    initData()
}
