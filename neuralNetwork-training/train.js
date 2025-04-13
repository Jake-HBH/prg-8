import posedata from "./data.json" with {type: 'json'};

const statusDiv = document.getElementById('status');
let nn;

// deze functie start het trainen
function startTraining() {
    // zet een bericht op het scherm
    statusDiv.textContent = 'Ai is aan het leren...';

    ml5.setBackend('webgl');

    // maak een nieuw ai model dat de poses herkent
    nn = ml5.neuralNetwork({task: 'classification', debug: true});
    console.log(nn);

    console.log(`Er worden ${posedata.length} poses toegevoegd`);

    // voeg elke pose toe aan het ai model
    for (let pose of posedata) {
        console.log(pose);

        const fullData = pose.data;

        // geef de pose en het juiste antwoord aan het model
        nn.addData(fullData, {label: pose.label});
    }

    nn.normalizeData();

    // laat de ai leren/trainen (100 keer)
    nn.train({epochs: 100}, finishedTraining);
}

// na het trainen finishedTraining()
function finishedTraining() {
    console.log("Ai is klaar met leren");

    // sla het model op (in downloads bij mij, model.json model.weights.bin model_meta.json)
    nn.save();

    // kies 1 pose uit om te testen
    let demoPose = posedata[63].data;

    // vraag aan de ai wat dit is
    nn.classify(demoPose, (results) => {
        console.log(`Ik denk dat deze pose is: ${results[0].label}`);
        console.log(`Ik ben ${results[0].confidence.toFixed(2) * 100}% zeker`);
    });
}

// start alles
startTraining();
