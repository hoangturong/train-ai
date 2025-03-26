const fs = require("fs").promises;

async function readDataFile(filePath) {
  try {
    const fileContent = await fs.readFile(filePath, "utf8");
    const dataObject = JSON.parse(fileContent);
    return dataObject;
  } catch (err) {
    console.log("Error reading data file:", err);
    throw new Error("Failed to read data file");
  }
}

module.exports = readDataFile;
