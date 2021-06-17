// 선형 회귀 예제 코드

const X = [1, 2, 3];
const Y = [3, 5, 7];
const alpha = 0.001; // 학습률
const epoch = 100000; // 학습 회수

/**
 * 가설함수
 */
function H(weight, bias) {
  return (x) => weight * x + bias;
}

/**
 * 비용함수. 손실함수(loss)라고도 부른다.
 * 최소 제곱법을 사용한다.
 */
function cost(weight, bias) {
  // 가설
  const _H = H(weight, bias);

  // 데이터의 개수
  const n = X.length;

  // 오차 제곱의 합
  let sum = 0;

  for (let index = 0; index < n; index++) {
    const x = X[index];
    const y = Y[index];
    sum += (_H(x) - y) ** 2;
  }

  // (오차 제곱의 합) / (데이터의 개수)  = 비용 = 손실
  return sum / n;
}

/**
 * 비용함수를 weight에 대하여 편미분한 함수
 */
function costD_weight(weight, bias) {
  const f = (w, b, x, y) => 2 * x ** 2 * w + 2 * x * b - 2 * x * y;

  // 데이터의 개수
  const n = X.length;
  let sum = 0;

  for (let index = 0; index < n; index++) {
    const x = X[index];
    const y = Y[index];
    sum += f(weight, bias, x, y);
  }

  return sum / n;
}

/**
 * 비용함수를 bias에 대하여 편미분한 함수
 */
function costD_bias(weight, bias) {
  const f = (w, b, x, y) => 2 * x * w - 2 * y + 2 * b;

  // 데이터의 개수
  const n = X.length;
  let sum = 0;

  for (let index = 0; index < n; index++) {
    const x = X[index];
    const y = Y[index];
    sum += f(weight, bias, x, y);
  }

  return sum / n;
}

module.exports = async function () {
  let weight = Math.random() * 10;
  let bias = Math.random() * 10;

  for (let index = 0; index < epoch; index++) {
    console.log("cost", cost(weight, bias));
    weight = weight - alpha * costD_weight(weight, bias);
    bias = bias - alpha * costD_bias(weight, bias);
  }

  console.log({ weight, bias });
};
