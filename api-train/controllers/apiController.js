const fs = require("fs").promises;
const markdownIt = require("markdown-it");

const rootController = async (req, res) => {
  try {
    const viPath = "docs/docs.vi.md";
    const enPath = "docs/docs.en.md";

    const [viData, enData] = await Promise.all([
      fs.readFile(viPath, "utf8"),
      fs.readFile(enPath, "utf8"),
    ]);

    const md = new markdownIt();
    const viContent = md.render(viData);
    const enContent = md.render(enData);

    const data = {
      vi: viContent,
      en: enContent,
    };

    res.json(data);
  } catch (error) {
    res.status(500).send("Internal Server Error");
  }
};

module.exports = rootController;
