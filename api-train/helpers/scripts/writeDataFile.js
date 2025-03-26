const fs = require("fs").promises;

async function writeDataFile(filePath, dataObject) {
  try {
    const jsonString = JSON.stringify(dataObject);
    await fs.writeFile(filePath, jsonString);
    console.log("Data file has been updated");
  } catch (err) {
    console.log("Error writing data file:", err);
    throw new Error("Failed to write data file");
  }
}

module.exports = writeDataFile;
