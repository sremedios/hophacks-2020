async function runExample() {

  // Create an ONNX inference session with default backend.
  const session = new onnx.InferenceSession();
  
  // Load model
  const model_url = "../models/covid_detect.onnx"
  await session.loadModel(model_url);
  
  const input_img = [
	new onnx.Tensor(
		new Float32Array(1 * 1 * 512 * 512).fill(1), 
		"float32", 
		[1, 1, 512, 512],
	),
  ];

  // Run model with Tensor inputs and get the result by output name defined in model.
  const outputMap = await session.run(input_img);
  const outputData = outputMap.values().next().value;

  console.log("Hello")

}