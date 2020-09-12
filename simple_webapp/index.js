
async function runExample() {
	
  console.log("Hello")
  // Create an ONNX inference session with default backend.
  const session = new onnx.InferenceSession();
  // Load an ONNX model. This model is Resnet50 that takes a 1*3*224*224 image and classifies it.
  await session.loadModel("../models/covid_detect.onnx");

  const x = new Float32Array(1 * 1 * 512 * 512).fill(1);
  const tensorX = new onnx.Tensor(x, 'float32', [1, 1, 512, 512]);

  // Run model with Tensor inputs and get the result by output name defined in model.
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('sum');

  console.log("Hello")

}
