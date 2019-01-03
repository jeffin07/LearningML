let mousexs = [];
let mouseys = [];
let w,b;

const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate);
function setup(){

	createCanvas(400,400);
	w = tf.variable(tf.scalar(random(1)));
	b = tf.variable(tf.scalar(random(1)));


}
function mousePressed(){

	
	let x = map(mouseX,0,width,0,1);
	let y = map(mouseY,0,height,1,0);
	mousexs.push(x);
	mouseys.push(y);

}
function loss(preds, labels){
	return preds.sub(labels).square().mean();
}
function predict(xs){
	const tensx = tf.tensor1d(xs);
	const pred_y = tensx.mul(w).add(b);
	// pred_y.print();
	// const pred_y = tensx.mul(w).add(b);
	return pred_y

}
function draw(){
	if (mousexs.length > 0){
		const ys = tf.tensor1d(mouseys);
		predict(mousexs);
		// console.log((mousexs));
		optimizer.minimize(()=> loss(predict(mousexs),ys));
	}
	
	background(0);
	stroke(255);
	strokeWeight(4);
	
	for (let i = 0;i < mousexs.length ; i++){
		let txs = map(mousexs[i],0,1, 0, width)
		let tys = map(mouseys[i],0, 1, height, 0);
		point(txs, tys)
		
		// console.log(predict(mousexs));
	}
	const lineX = [0,1]

	ys = predict(lineX);
	let lineY = ys.dataSync();

	let x1 = map(lineX[0],0,1,0,width);
	let x2 = map(lineX[1],0,1,0,width);

	let y1 = map(lineY[0],0,1,height,0);
	let y2 = map(lineY[1],0,1,height,0);

	strokeWeight(2);
	line(x1,y1,x2,y2);

}