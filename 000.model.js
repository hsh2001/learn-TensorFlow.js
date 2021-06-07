const tf = require("@tensorflow/tfjs");

function expectedFunction(x) {
  return x * 2;
}

module.exports = async () => {
  const x = [20, 21, 22, 23];
  const y = x.map(expectedFunction);

  console.log({ x, y });

  const inputs = tf.input({ shape: [1] });
  const outputs = tf.layers.dense({ units: 1 }).apply(inputs);
  const model = tf.model({ inputs, outputs });

  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
  });

  await model.fit(tf.tensor(x), tf.tensor(y), {
    epochs: 20000,
    callbacks: {
      onEpochEnd(epochs, logs) {
        console.log({ epochs, RMSE: Math.sqrt(logs.loss) });
      },
    },
  });

  const input = 28;
  const [[output]] = await model.predict(tf.tensor([input])).array();
  const [weightTensor, biasTensor] = model.getWeights();
  const [[weight]] = await weightTensor.array();
  const [bias] = await biasTensor.array();

  console.log({
    input,
    output,
    looksLike: `y = ${weight} * x + ${bias}`,
    diff: Math.abs(expectedFunction(input) - output),
  });
};
