var express = require("express");
var bodyParser = require("body-parser");
// const cv2 = require('opencv4nodejs');
// var NodeWebcam = require( "node-webcam" );
const request = require('request');

const app = express();

app.set('view engine', 'ejs');

app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static("public"));

const url_endlessRunner_1 = "https://scratch.mit.edu/projects/600560206/embed"
const url_endlessRunner_2 = "https://scratch.mit.edu/projects/599708735/embed"
const url_endlessRunner_3 = "https://scratch.mit.edu/projects/600097774/embed"
var url_send=""
gameName=""

app.get("/", function (req, res) {
 
    res.render("index");

});

app.post("/", function (req, res) {

    console.log(req.body);
    var play=req.body.play;

    if(play=="yes"){
       
        // console.log("hi")
        res.redirect("/yoga")
    }        

})


app.get("/yoga", function (req, res) {
 
    // console.log("hi pt2")
    res.render("yoga");

});


app.get("/games", function (req, res) {
 
    res.render("games");

});

app.post("/games", function (req, res) {

    console.log(req.body);
    var gameName=req.body.gameName;

    if(gameName=="endless-runner-1"){
        url_send=url_endlessRunner_1;
        res.redirect("/play")
    } 
    else if(gameName=="endless-runner-2"){
        url_send=url_endlessRunner_2;
        res.redirect("/play")
    }
    else if(gameName=="endless-runner-3"){
        url_send=url_endlessRunner_3;
        res.redirect("/play")
    }

})



app.get("/play", function (req, res) {
 
    // const Vcap = new cv2.VideoCapture(0);

    // request('http://localhost:5000/flask', function (error, response, body) {
    //     console.error('error:', error); // Print the error
    //     console.log('statusCode:', response && response.statusCode); // Print the response status code if a response was received
    //     console.log('body:', body); // Print the data received
    //     // res.send(body); //Display the response on the website
    //   });

    // var Webcam = NodeWebcam.create({});
    res.render("play",{gameSrc:url_send});

});

// app.post("/play", function (req, res) {

//     console.log(req.body);
//     var gameName=req.body.gameName;

//     if(gameName=="endless-runner-1"){
//         res.redirect("/play");
//     }

// })


// host=process.env.PORT

app.listen(process.env.PORT || 3000, function () {

    console.log("Server is up and running on port 3000");

});

