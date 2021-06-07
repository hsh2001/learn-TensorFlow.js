const tf = require('@tensorflow/tfjs');

// module.exports =
async () => {
  // 여러개의 독립변수, 하나의 종속변수
  const { x, y } = require('./data/boston.json');
  const inputs = tf.input({ shape: [13] });
  const outputs = tf.layers.dense({ units: 1 }).apply(inputs);
  const model = tf.model({ inputs, outputs });

  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
  });

  await model.fit(tf.tensor(x), tf.tensor(y), {
    epochs: 100,
    callbacks: {
      onEpochEnd(epochs, logs) {
        console.log({ epochs, RMSE: Math.sqrt(logs.loss) });
      },
    },
  });

  model
    .predict(
      tf.tensor([
        [
          0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9,
          4.98,
        ],
      ]),
    )
    .print();
};

module.exports = async () => {
  // 여러개의 독립변수, 여러개의 종속변수
  const { x, y } = require('./data/boston2.json');
  const inputs = tf.input({ shape: [12] });
  const outputs = tf.layers.dense({ units: 2 }).apply(inputs);
  const model = tf.model({ inputs, outputs });

  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
  });

  await model.fit(tf.tensor(x), tf.tensor(y), {
    epochs: 100,
    callbacks: {
      onEpochEnd(epochs, logs) {
        console.log({ epochs, RMSE: Math.sqrt(logs.loss) });
      },
    },
  });

  model
    .predict(
      tf.tensor([
        [0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9],
      ]),
    )
    .print();
};
