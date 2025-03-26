const express = require("express");
const resController = require("../controllers/resController");
const apiController = require("../controllers/apiController.js");
const router = express.Router();
router.get("/api", apiController);
router.get("/game/:number", resController.index);
router.get("/data/:number", resController.data);
module.exports = router;
