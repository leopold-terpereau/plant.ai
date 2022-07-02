
// Récupération de l'image fournie par l'utilisateur et affichage de cette dernière 
$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL =reader.result;
        $('#selected-image').attr("src", dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});


// Récupération de la valeur de l'onglet de sélection de l'espèce de plante considérée, et chargement du modèle prédictif associé
$("#model-selector").change(function () {
    loadModel($("#model-selector").val())
});


// Définition de la fonction de chargement de modèle, et affichage d'une barre de progression
let model;
async function loadModel(name) {
    $(".progress-bar").show();
    model = undefined;
    model = await tf.loadLayersModel("./tfjs-models/"+name+"/model.json"); // Le modèle prédictif est récupéré dans le dossier tfjs-models et est choisi selon l'espèce sélectionnée
    $(".progress-bar").hide();
}

// Fonction asynchrone de diagnostic (ou prédiction)
$("#predict-button").click(async function () {
    let image = $("#selected-image").get(0);                    // stockage de la variable contenant l'image fournie par l'utilisateur
    let modelName = $("#model-selector").val();                 // récupération de la valeur du modèle càd de l'espèce sélectionnée
    let tensor = tf.browser.fromPixels(image)                   // déclaration du tenseur qui contient l'image (dimension (1,n,m,3))
        .resizeNearestNeighbor([256, 256])                      // redimmensionnement de l'image en (1,256,256,3) 
        .toFloat()
        .expandDims();



    // Si le modèle sélectionné est la recconnaissance de l'espèce, on recupère les sorties possibles du modèle dans SPECIES_CLASSES
    if (modelName == "Reconnaissance de l'espèce"){
        let predictions = await model.predict(tensor).data();   // On applique le CNN au tenseur et on récupère la sortie
        let result = Array.from(predictions)
        .map(function (p,i) {
            return {
                probability: p, 
                className: SPECIES_CLASSES[i]
            };
        })
        .sort(function (a, b) {
            return b.probability - a.probability;
       }).slice(0, 1);
        $("#prediction-list").empty();
        result.forEach(function (p) {   
            $('#prediction-list').append(`<p> Work in progress 
            <br>
            <b> ${p.className}  </b> </p>`);                    // On affiche l'espèce reconnue par le CNN
        });
    }

                                                                // Si le modèle sélectionné est un nom d'espèce, on recupère les sorties possibles du modèle dans HEALTH_CLASSES
    else{
        const offset = tf.scalar(255);  
        tensor = tensor.div(offset);                                // normalisation du tenseur càd division des valeurs des pixels (entre 0 et 255) par 255
        let predictions = await model.predict(tensor).data();   // On applique le CNN au tenseur et on récupère la sortie
        let result = Array.from(predictions)
        .map(function (p,i) {                                   // On associe à chaque sortie (càd à chaque valeur de probabilité) la classe (càd l'état de santé) correspondant
            return {
                probability: p, 
                className: HEALTH_CLASSES[i] 
            };
        }).sort(function (a, b) {
            return b.probability - a.probability;               // On trie les sorties par probabilité décroissante
        }).slice(0, 1);                                         // On récupère la plus grande probabilité et l'état de santé associé
        $("#prediction-list").empty();
        result.forEach(function (p) {
            if (p.className=="La plante est en mauvaise santé") {
                $('#prediction-list').append(`<p style="color:#FF0000;text-align:left";> <b>${p.className} </b></p>`); // On affiche "La plante est en mauvaise santé" en rouge
            }
            else{
                $('#prediction-list').append(`<p style="color:#228b22;text-align:left";> <b>${p.className} </b></p>`); // On affiche "La plante est en bonne santé" en vert
            }
        });
    }

}); 
