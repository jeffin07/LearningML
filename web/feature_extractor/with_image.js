var catimage = document.getElementById('cat');
var dogimage = document.getElementById('dog');
var owlimage = document.getElementById('owl');
console.log(dogimage);
console.log(catimage);
console.log(owlimage)
var classifier = '';

function modelLoaded(){
	console.log('modelLoaded')
	classifier = model.regression(adding_image);
console.log(classifier)
// classifier.addImage(catimage,"cat",adding_image)
// classifier.addImage(catimage,"cat",adding_image)
// classifier.addImage(catimage,"cat",adding_image)


// classifier.addImage(dogimage,"dog",adding_image)
// classifier.addImage(dogimage,"dog",adding_image)
// classifier.addImage(dogimage,"dog",adding_image)



// classifier.train(training)




}
function addcat(){
	console.log("adding Cat")
	classifier.addImage(catimage,0,adding_image)
}
function adddog(){
	console.log("adding Dog")
	classifier.addImage(dogimage,1,adding_image)
}
function addowl(){
	console.log("adding owl")
	classifier.addImage(owlimage,2,adding_image)
}
function trainer(){
	classifier.train(training)
}
function adding_image(){
	console.log("added")
}
function gotresults(err,data){
	var names = ["cat","dog","owl"]
	if (err) {
		console.log(err)
	}
	else{
		console.log("Perdiction for the given image is : "+names[Math.round(data)])
	}
}
function training(loss){
	if(loss == null){
		console.log("training complete")
				classifier.predict(catimage,gotresults)
		classifier.predict(dogimage,gotresults)
		classifier.predict(owlimage,gotresults)

	}
	else{
		console.log("loss : "+loss)
	}

}
var model = ml5.featureExtractor('MobileNet',modelLoaded);
 
