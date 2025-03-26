function rand(min, max, shold = 0.5) {
  const totalOptions = max - min + 1;
  const randomValue = Math.random();
  console.log(randomValue, shold);
  if (randomValue < shold) {
    const range = Math.floor(totalOptions / 2);
    return Math.floor(randomValue * range) + min;
  } else {
    const range = Math.ceil(totalOptions / 2);
    return Math.floor(randomValue * range) + max - range + 1;
  }
}

module.exports = rand;
