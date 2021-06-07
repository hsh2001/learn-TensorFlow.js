const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const modelPath = `file://${__dirname}/model/times2`;

async function fitModel(model) {
  const x = [20, 21, 22, 23];
  const y = x.map((x) => x * 2);
  await model.fit(tf.tensor(x), tf.tensor(y), {
    epochs: 100,
    callbacks: {
      // onEpochEnd(epochs, logs) {
      //   console.log({ epochs, RMSE: Math.sqrt(logs.loss) });
      // },
    },
  });
}

async function createModel() {
  const inputs = tf.input({ shape: [1] });
  const outputs = tf.layers.dense({ units: 1 }).apply(inputs);
  const model = tf.model({ inputs, outputs });

  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
  });

  await fitModel(model);

  return model;
}

async function saveModel(model) {
  await model.save(modelPath);
}

async function loadModel() {
  return await tf.loadLayersModel(`${modelPath}/model.json`);
}

module.exports = async () => {
  // saveModel(await createModel());
  const model = await loadModel();
  model.predict(tf.tensor([20])).print();
};
