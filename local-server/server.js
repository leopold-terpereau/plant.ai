// Déclaration des dépendances
let express = require("express");
let app = express();

// Affichage des requêtes du serveur local
app.use(function(req, res, next) {
    console.log(`${new Date()} - ${req.method} request for ${req.url}`);
    next();
});

// Déclaration du dossier static contenant les modules back et front end de l'application
app.use(express.static("../static"));

// Instanciation du serveur local, sur le port 3
app.listen(3, function() {
    console.log("L'application fonctionne, allez sur localhost:3/main.html");
})