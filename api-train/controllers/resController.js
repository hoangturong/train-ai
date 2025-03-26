const path = require("path");
const dataFilePath = path.join(__dirname, "../data/data.json");
const { rand, readDataFile, writeDataFile } = require("../helpers");

module.exports = {
  async index(req, res) {
    const { number } = req.params;
    if (!number || isNaN(number)) {
      return res.status(400).json({
        message: "Please enter a valid number",
      });
    }
    const parsedNumber = parseInt(number);

    try {
      const dataObject = await readDataFile(dataFilePath);

      const taiCount = dataObject.data.filter(
        (item) => item.type === "Tài"
      ).length;
      const xiuCount = dataObject.data.filter(
        (item) => item.type === "Xỉu"
      ).length;
      const total = taiCount + xiuCount;
      const currentRatio = total > 0 ? taiCount / total : 0;
      // Điều chỉnh threshold dựa trên tỷ lệ hiện tại
      const threshold = currentRatio - 0.1;
      const data = [];
      let response = 0;

      for (let i = 1; i <= parsedNumber; i++) {
        const random = rand(1, 6, threshold);
        response += random;
        data.push(random);
      }

      const result = {
        response,
        data,
        type: "",
      };

      if (data.every((value) => value === data[0])) {
        result.type = "Bộ";
      } else if (
        response >= parsedNumber + 1 &&
        response <= parsedNumber * parsedNumber + 1
      ) {
        result.type = "Xỉu";
      } else if (
        response >= parsedNumber * parsedNumber + 2 &&
        response <= parsedNumber * (parsedNumber * 2) - 1
      ) {
        result.type = "Tài";
      } else {
        result.type = "Đặc biệt";
      }

      dataObject.data.push(result);

      await writeDataFile(dataFilePath, dataObject);

      res.json(result);
    } catch (err) {
      res.status(500).json({
        message: "Failed to read/write data file",
      });
    }
  },

  async data(req, res) {
    const { number } = req.params;
    if (!number || isNaN(number)) {
      return res.status(400).json({
        message: "Please enter a valid number",
      });
    }
    const parsedNumber = parseInt(number);

    try {
      const dataObject = await readDataFile(dataFilePath);
      const history = dataObject.data.slice(-parsedNumber);
      res.json(history);
    } catch (err) {
      res.status(500).json({
        message: "Failed to read data file",
      });
    }
  },
};
