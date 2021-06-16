const tf = require("@tensorflow/tfjs");

//             rank 0 tensor === scalar
//             rank 1 tensor === vector
//             rank 2 tensor === matrix
//                 .
//                 .
//                 .
//             rank n tensor === n tensor

module.exports = async () => {
  //    스칼라 tensor
  //    rank = 0
  const one = tf.scalar(1);
  const two = tf.scalar(2);
  const five = tf.scalar(5);

  const maybeThree = tf.add(one, two);
  maybeThree.print();
  //   Tensor
  //     3

  const maybeZeroPointTwo = one.div(five);
  maybeZeroPointTwo.print();
  //   Tensor
  //     0.20000000298023224

  ////////////////////////////////////////

  //    벡터 tensor
  //    rank = 1
  const vector1 = tf.tensor1d([1, 3, 5]);
  const vector2 = tf.tensor1d([2, 3, 4]);

  // 두 벡터의 연산은 같은 위치에 있는 요소끼리 더하는 것으로 한다.
  // x = (x1, x2, ... xn)
  // y = (y1, y2, ... yn)
  // x + y = (x1 + y1, x2 + y2 ... xn + yn)
  // (1,2,3) + (2,3,4) = (3,5,7)

  vector1.add(vector2).print();
  // Tensor
  //     [3, 6, 9]

  const vector3 = tf.tensor1d([1, 23, 3]);
  const scalar = tf.scalar(3);

  // 벡터 + 스칼라 => 벡터 브로드캐스팅
  // 벡터의 각 요소에 스칼라 값을 더한다.
  // 벡터 v = (x,y,z)
  // 스칼라 s = a
  // v + s = (x + a, y + a, z + a)
  vector3.add(scalar).print();
  // Tensor
  //   [4, 26, 6]

  const v1 = tf.tensor1d([1, 2, 3]);
  const v2 = tf.tensor1d([4, 5, 6]);

  // 벡터의 내적
  // 벡터 x, y에 대하여,
  // x = (x1, x2, ... xn)
  // y = (y1, y2, ... yn)
  // 일때, 두 벡터의 내적 <x, y> = x1 * y1 + x2 * y2 + ... + xn + yn
  // 즉, 같은 자리의 요소끼리 곱한 스칼라 값들을 더한 값이다.
  // 벡터의 내적은 x · y 로도 표현한다.
  v1.dot(v2).print();
  // Tensor
  //   32

  // 벡터의 내적
  // 벡터 x, y에 대하여,
  // x = (x1, x2, ... xn)
  // y = (y1, y2, ... yn)
  // 일때, 두 벡터의 내적 x ⊗ y =
  // | x1 * y1 + x1 * y2 + ... +  x1 * yn  |
  // | x2 * y1 + x2 * y2 + ... +  x2 * yn  |
  // | ................................... |
  // | xn * y1 + xn * y2 + ... +  xn * yn  |
  tf.outerProduct(v1, v2);
};
