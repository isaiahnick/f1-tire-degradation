// PitWall final presentation (EM 538, Team 9). Run: node build.js
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const NOTES = require("./notes.js");
const { applyTheme } = require("/root/.claude/skills/synced/a69a8812-da28-4575-95a4-32b6eec2f8d8_4226a6e8-952f-4a4b-be1f-bea8c45734ee/pptx/scripts/apply_theme.js");

const FIG = "/home/claude/pw_laptop/reports/figures/slides/";
const OUT = "/home/claude/pptx_build/PitWall_Final_Presentation.pptx";

const RED = "CC0000", INK = "1A1A1A", GREY = "595959", MIDG = "8C8C8C", LIGHT = "F2F2F2", DARK = "404040";
const THEME = {
  name: "PitWall", headFontFace: "Arial", bodyFontFace: "Arial",
  colors: { dk1: "1A1A1A", lt1: "FFFFFF", dk2: "404040", lt2: "F2F2F2",
    accent1: "CC0000", accent2: "404040", accent3: "8C8C8C", accent4: "C8C8C8",
    accent5: "E8B400", accent6: "7F7F7F", hlink: "0563C1", folHlink: "954F72" },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "PitWall: Predicting F1 Tire Degradation to Recommend Pit Stop Timing";
pres.author = "Isaiah Nick, Anand Vadlamani";
pres.subject = "EM 538 Final Project Presentation";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };

const bandObjs = (h, right) => [
  { rect: { x: 0, y: 0, w: 10, h, fill: { color: RED }, line: { color: RED } } },
  { text: { text: "NC STATE UNIVERSITY", options: { x: 0.4, y: 0, w: 4, h, fontSize: 11, bold: true,
    color: "FFFFFF", valign: "middle", margin: 0 } } },
  ...(right ? [{ text: { text: right, options: { x: 5.6, y: 0, w: 4.0, h, fontSize: 10, color: "FFFFFF",
    align: "right", valign: "middle", margin: 0 } } }] : []),
];
pres.defineSlideMaster({
  title: "DARK", background: { color: "1A1A1A" },
  objects: [
    ...bandObjs(0.42),
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.0, w: 8.8, h: 1.5,
      fontSize: 32, bold: true, align: "left", valign: "bottom", color: "FFFFFF" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.75, w: 5.6, h: 1.4,
      fontSize: 15, align: "left", valign: "top", color: "BFBFBF" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  slideNumber: { x: 9.2, y: 5.28, w: 0.5, h: 0.25, fontSize: 10, color: MIDG, align: "right" },
  objects: [
    ...bandObjs(0.3, "EM 538 Final Project  |  Team 9"),
    { placeholder: { options: { name: "title", type: "title", x: 0.45, y: 0.42, w: 9.1, h: 0.6,
      fontSize: 24, bold: true, align: "left", valign: "middle", color: INK }, text: "" } },
  ],
});

// ---------------------------------------------------------------- helpers
function pngSize(file) { const b = fs.readFileSync(file); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
function slideWith(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function img(s, name, x, y, w) {
  const [pw, ph] = pngSize(FIG + name);
  s.addImage({ path: FIG + name, x, y, w, h: w * ph / pw, altText: name });
  return y + w * ph / pw;
}
function bullets(s, items, x, y, w, h, size = 15, gap = 7) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, fontSize: size,
    breakLine: i < items.length - 1, paraSpaceAfter: gap } }));
  s.addText(runs, { x, y, w, h, valign: "top", color: INK, isTextBox: true, margin: 2 });
}
function text(s, t, x, y, w, h, o = {}) {
  s.addText(t, { x, y, w, h, fontSize: 13, color: GREY, valign: "top", isTextBox: true, margin: 2, ...o });
}
function head(s, t, x, y, w, color = INK) {
  s.addText(t, { x, y, w, h: 0.32, fontSize: 14, bold: true, color, valign: "bottom", isTextBox: true, margin: 2 });
}
function stat(s, num, label, x, y, w) {
  s.addText([{ text: num, options: { fontSize: 40, bold: true, color: RED, breakLine: true } },
    { text: label, options: { fontSize: 13, color: GREY } }], { x, y, w, h: 1.15, valign: "top", isTextBox: true, margin: 2 });
}
function table(s, rows, x, y, w, colW, size = 12, hilite = [], rowH = 0.29) {
  const data = rows.map((r, i) => r.map((c, j) => {
    const cell = typeof c === "object" ? c : { text: String(c) };
    return { text: cell.text, options: {
      bold: i === 0 || hilite.includes(i) || cell.bold, color: i === 0 ? "FFFFFF" : (cell.color || INK),
      fill: { color: i === 0 ? DARK : (hilite.includes(i) ? "FBE5E5" : (i % 2 === 0 ? "F7F7F7" : "FFFFFF")) },
      align: j === 0 ? "left" : "center", fontSize: size } };
  }));
  s.addTable(data, { x, y, w, colW, rowH, border: { type: "solid", pt: 0.5, color: "D0D0D0" }, valign: "middle", margin: 0.06 });
}
// one-line takeaway in a light grey box at the bottom of the slide
function takeaway(s, t, y = 4.45, h = 0.65) {
  s.addShape(pres.shapes.RECTANGLE, { x: 0.45, y, w: 9.1, h, fill: { color: LIGHT }, line: { color: LIGHT } });
  s.addText(t, { x: 0.6, y, w: 8.8, h, fontSize: 13, color: INK, valign: "middle", isTextBox: true, margin: 2 });
}
// tire sidewall: rubber, compound-colored ring, grey rim
function tire(s, x, y, d, ring) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: "0D0D0D" }, line: { color: "3A3A3A", width: 1 } });
  const r = d * 0.13;
  s.addShape(pres.shapes.OVAL, { x: x + r, y: y + r, w: d - 2 * r, h: d - 2 * r, fill: { color: "0D0D0D" },
    line: { color: ring, width: Math.max(2, d * 7) } });
  const q = d * 0.33;
  s.addShape(pres.shapes.OVAL, { x: x + q, y: y + q, w: d - 2 * q, h: d - 2 * q, fill: { color: "A6A6A6" },
    line: { color: "7F7F7F", width: 1 } });
}
function tires(s, x0, y, d, gap = 0.25) {
  ["E10600", "FFD100", "F2F2F2"].forEach((c, i) => tire(s, x0 + i * (d + gap), y, d, c));
}

// ================================================================ 1. Title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
s.addText("PitWall: Predicting F1 Tire Degradation to Recommend Pit Stop Timing", { placeholder: "title" });
s.addText("Team 9: Isaiah Nick, Anand Vadlamani\nEM 538 Final Project, Fall 2026\nInstructor: Dr. Fred Livingston", { placeholder: "body" });
tires(s, 6.55, 3.75, 0.85, 0.2);
s.addNotes(NOTES[0]);

// ================================================================ 2. Problem and objectives
s = slideWith("Problem and Objectives", "Introduction");
bullets(s, [
  "Tires lose grip with every lap, so each lap on a set gets slower",
  "A pit stop costs about 20 s; stopping too early or too late loses race time",
  "Users: race strategists and tire engineers",
  "ML task: predict lap time loss (s) for the rest of a stint, then recommend a pit lap",
], 0.45, 1.2, 4.3, 2.7, 15);
img(s, "s_stint_example.png", 4.95, 1.12, 4.7);
text(s, "2025 Spanish GP, Verstappen. Lap time climbs until each pit stop.", 4.95, 3.86, 4.7, 0.28, { fontSize: 10, color: MIDG });
[["Goal 1: prediction", "Test MAE at least 15% below a linear wear baseline, on a season the models never saw (2025)"],
 ["Goal 2: decision", "Recommended pit lap within 3 laps of the team's actual stop for 60% of green-flag stops"]]
  .forEach(([h, b], i) => {
    const x = 0.45 + i * 4.6;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 4.2, w: 4.45, h: 0.9, fill: { color: LIGHT }, line: { color: LIGHT } });
    s.addText([{ text: h, options: { bold: true, color: RED, fontSize: 13, breakLine: true } },
      { text: b, options: { color: INK, fontSize: 12 } }], { x: x + 0.12, y: 4.25, w: 4.2, h: 0.8, valign: "top", isTextBox: true, margin: 2 });
  });
s.addNotes(NOTES[1]);

// ================================================================ 3. Data
pres.addSection({ title: "Data and Features" });
s = slideWith("Data and Preprocessing", "Data and Features");
bullets(s, [
  "FastF1 timing data for every race from 2022 to 2025, plus practice sessions (pinned archive)",
  "Target: how much slower a lap is than the first 3 laps of its stint, after removing the fuel effect",
  "Train: 2022-2024 (58 races)   Test: 2025 (20 races)",
], 0.45, 1.15, 4.4, 1.8, 13, 5);
const funnel = [["Raw laps", 101297], ["Wet races, Monaco removed", 86014], ["Lap 1, pit laps, safety car, rain removed", 72759],
  ["Outliers removed", 69615], ["Target laps (stints of 6+ clean laps)", 57813]];
funnel.forEach(([lab, n], i) => {
  const y = 3.0 + i * 0.4, last = i === funnel.length - 1;
  text(s, lab, 0.45, y, 2.3, 0.34, { fontSize: 11, color: last ? INK : GREY, bold: last, valign: "middle" });
  const w = 1.55 * n / 101297;
  s.addShape(pres.shapes.RECTANGLE, { x: 2.8, y: y + 0.05, w, h: 0.24, fill: { color: last ? RED : "A6A6A6" }, line: { color: last ? RED : "A6A6A6" } });
  text(s, n.toLocaleString("en-US"), 2.85 + w, y, 0.9, 0.34, { fontSize: 11, color: INK, bold: last, valign: "middle" });
});
head(s, "Lap time loss by compound (training data)", 5.0, 1.1, 4.6);
img(s, "s_degradation.png", 5.0, 1.47, 4.6);
text(s, "Softs wear fastest, then mediums, then hards. Shaded: 95% CI.", 5.0, 4.15, 4.6, 0.3, { fontSize: 11, color: MIDG });
s.addNotes(NOTES[2]);

// ================================================================ 4. Features
s = slideWith("Feature Development", "Data and Features");
const seg = [["Laps 1-3", "reference pace", "D9D9D9", INK, 2.2], ["Laps 4-8", "seen by the pit wall", DARK, "FFFFFF", 3.0],
  ["Laps 9 onward", "predicted", "FBE5E5", RED, 3.9]];
let sx = 0.45;
seg.forEach(([a, b, fill, col, w]) => {
  s.addShape(pres.shapes.RECTANGLE, { x: sx, y: 1.15, w, h: 0.62, fill: { color: fill }, line: { color: fill === "FBE5E5" ? RED : fill, width: 1 } });
  s.addText([{ text: a, options: { bold: true, fontSize: 13, color: col, breakLine: true } }, { text: b, options: { fontSize: 11, color: col } }],
    { x: sx + 0.1, y: 1.17, w: w - 0.2, h: 0.58, valign: "middle", isTextBox: true, margin: 2 });
  sx += w;
});
head(s, "Setup A: before the stint starts", 0.45, 1.95, 4.4);
bullets(s, [
  "Compound, laps on the set, tire age, new or used set",
  "Laps remaining, stint number, position, gap to car ahead",
  "Circuit and team",
  "Practice long-run wear rate for that weekend (new)",
], 0.45, 2.3, 4.4, 2.0, 13, 4);
head(s, "Setup B: after 5 laps on the set", 5.15, 1.95, 4.4, RED);
bullets(s, [
  "Everything in Setup A, plus:",
  "Mean loss and wear slope over the first 5 laps",
  "Time in dirty air and mean gap ahead (new)",
  "Car pace vs. the field on the same laps (new)",
], 5.15, 2.3, 4.4, 2.0, 13, 4);
takeaway(s, "Each change was tested with cross-validation: new features lowered error about 4%, adding 2022 about 3%. Weather was dropped (no gain).");
s.addNotes(NOTES[3]);

// ================================================================ 5. Models and validation
pres.addSection({ title: "Methods and Results" });
s = slideWith("Models and Validation", "Methods and Results");
[["2022", "D9D9D9", INK], ["2023", "D9D9D9", INK], ["2024", "D9D9D9", INK], ["2025", RED, "FFFFFF"]].forEach(([yr, fill, col], i) => {
  const x = 0.45 + i * 2.3;
  s.addShape(pres.shapes.RECTANGLE, { x, y: 1.15, w: 2.2, h: 0.42, fill: { color: fill }, line: { color: fill } });
  s.addText(yr, { x, y: 1.15, w: 2.2, h: 0.42, fontSize: 14, bold: true, color: col, align: "center", valign: "middle", isTextBox: true, margin: 0 });
});
text(s, "Training: 5-fold cross-validation grouped by race, grid search inside the folds", 0.45, 1.6, 6.8, 0.3, { fontSize: 11 });
text(s, "Test: held out until the end", 7.35, 1.6, 2.2, 0.3, { fontSize: 11, color: RED, align: "center" });
table(s, [
  ["Model", "Why it is included", "Key settings"],
  ["Linear regression", "Baseline: one wear rate per compound", "-"],
  ["Ridge", "Regularized linear model, easy to explain", "alpha = 1000"],
  ["Decision tree", "Simple nonlinear model", "depth 8, min leaf 100"],
  ["Random forest", "Bagging ensemble of trees", "200 trees, min leaf 20"],
  ["Gradient boosting", "Boosting ensemble of trees", "learning rate 0.03, 300 trees"],
  ["LSTM (PyTorch)", "Reads the first 5 laps as a sequence", "32 units, dropout 0.2"],
  ["GB + Ridge average", "Final model, chosen by CV; updated after every lap", "equal weights"],
], 0.45, 2.05, 9.1, [2.2, 4.3, 2.6], 12, [7]);
text(s, "Scaling and one-hot encoding are fit inside each fold with scikit-learn Pipelines. Metric: MAE in seconds per lap.", 0.45, 4.6, 9.1, 0.35, { fontSize: 11 });
s.addNotes(NOTES[4]);

// ================================================================ 6. Evaluation
s = slideWith("Model Evaluation (Setup B, 2025 Test Season)", "Methods and Results");
img(s, "s_train_test.png", 0.35, 1.12, 5.6);
stat(s, "34%", "lower test MAE than the linear baseline (goal: 15%)", 6.2, 1.12, 3.4);
bullets(s, [
  "Average miss per lap: 0.41 s (baseline: 0.63 s)",
  "Trees overfit: train 0.28 s, test 0.43 s",
  "Baseline underfits: R² 0.06",
  "Conditions only (Setup A): 5% better",
], 6.2, 2.4, 3.4, 2.0, 13, 6);
text(s, "For scale: a soft tire is about 1 s slower after 20 laps.", 6.2, 4.45, 3.4, 0.5, { fontSize: 11, color: MIDG });
s.addNotes(NOTES[5]);

// ================================================================ 7. Interpretation
s = slideWith("Where the Model Works and Where It Fails", "Methods and Results");
head(s, "What drives the prediction", 0.45, 1.1, 4.5);
img(s, "s_importance.png", 0.45, 1.45, 4.35);
head(s, "Forecast made after lap 5: error by distance", 5.15, 1.1, 4.5);
img(s, "s_horizon.png", 5.2, 1.45, 4.35);
takeaway(s, "Error grows the further ahead we predict: 0.31 s per lap within 5 laps, 0.60 s past 20. Hardest cases: street circuits and hard tires (longest stints).", 4.5, 0.6);
s.addNotes(NOTES[6]);

// ================================================================ 8. Forecast update
s = slideWith("Updating the Forecast After Every Lap", "Methods and Results");
stat(s, "24%", "lower error after lap 5 (0.38 vs. 0.50 s per lap)", 0.45, 1.15, 4.1);
bullets(s, [
  "Same model as before; stops the error growth from the last slide",
  "After each lap, shift the forecast by its average error on the last 3 laps",
  "Uses only completed laps; no retraining",
  "Part of the final model and the dashboard",
], 0.45, 2.45, 4.1, 2.6, 13, 7);
head(s, "GB + Ridge ensemble, forecast re-made each lap", 4.8, 1.1, 4.8);
img(s, "s_update.png", 4.8, 1.47, 4.8);
s.addNotes(NOTES[7]);

// ================================================================ 9. Pit stop recommendation
pres.addSection({ title: "Application" });
s = slideWith("From Tire Predictions to Pit Decisions", "Application");
head(s, "Tire model: when is it cheapest to stop?", 0.45, 1.1, 4.2, RED);
text(s, "Picks the lap with the least predicted time lost on old plus new tires: the tire-optimal pit window.", 0.45, 1.45, 4.2, 0.7, { fontSize: 13, color: INK });
head(s, "Team-stop model: when do teams stop?", 0.45, 2.25, 4.2, RED);
text(s, "Boosting model trained on 1,230 real stops (2022-2024) from race context: position, gap behind, stint, laps left.", 0.45, 2.6, 4.2, 0.9, { fontSize: 13, color: INK });
img(s, "s_pit_backtest.png", 4.8, 1.15, 4.8);
takeaway(s, "Team timing follows strategy we cannot fully observe. PitWall supports the call: it shows the tire-optimal window next to what teams typically do.", 4.4, 0.65);
s.addNotes(NOTES[8]);

// ================================================================ 10. Demo
s = slideWith("Demonstration: PitWall Dashboard", "Application");
const [dw, dh] = pngSize(FIG + "s_dashboard.png");
const DW = 6.6;
s.addShape(pres.shapes.RECTANGLE, { x: 0.43, y: 1.1, w: DW + 0.04, h: DW * dh / dw + 0.04, fill: { color: "FFFFFF" },
  line: { color: "BFBFBF", width: 1 }, shadow: { type: "outer", blur: 6, offset: 2, angle: 90, color: "000000", opacity: 0.18 } });
img(s, "s_dashboard.png", 0.45, 1.12, DW);
head(s, "Live demo", 7.3, 1.1, 2.3, RED);
bullets(s, [
  "Pick a 2025 race, driver and stint",
  "Tire-optimal window, typical team call, actual stop",
  "Slide the decision lap",
  "Switch models",
  "Plan a stint",
], 7.3, 1.48, 2.3, 2.8, 13, 7);
text(s, "One recommendation: 0.03 to 0.35 s", 7.3, 4.5, 2.3, 0.4, { fontSize: 11 });
s.addNotes(NOTES[9]);

// ================================================================ 11. Limitations
pres.addSection({ title: "Conclusion" });
s = slideWith("Limitations and Future Work", "Conclusion");
head(s, "Limitations", 0.45, 1.1, 4.4, RED);
bullets(s, [
  "Public timing only: no tire temperatures, fuel loads or team plans",
  "Fuel effect is an assumption (full tank, linear burn)",
  "Backtest measures agreement with teams, not the fastest possible race",
  "Needs 8 clean laps into a stint; wet races and Monaco excluded",
  "2025 was scored before and after our final changes",
], 0.45, 1.5, 4.4, 3.6, 13, 6);
head(s, "Future work", 5.15, 1.1, 4.4, RED);
bullets(s, [
  "Test on 2026 cars, with narrower tires and new rules",
  "Judge pit calls with a race-time simulation",
  "Calibrated prediction intervals",
  "Add safety car risk and planned stop counts to the pit window",
], 5.15, 1.5, 4.4, 3.6, 13, 6);
s.addNotes(NOTES[10]);

// ================================================================ 12. Conclusions
s = slideWith("Conclusions", "Conclusion");
table(s, [
  ["Goal", "Target", "Result on 2025", "Met"],
  ["Predict lap time loss", "15% below baseline", "34% below (0.412 s vs. 0.628 s)", { text: "Yes", color: "2E7D32", bold: true }],
  ["Pit lap within 3 laps of actual stop", "60% of stops", "25% (tire model), 56% (team-stop model)", { text: "No", color: RED, bold: true }],
  ["Fast enough for the pit wall", "under 1 s", "0.03 to 0.35 s", { text: "Yes", color: "2E7D32", bold: true }],
], 0.45, 1.15, 9.1, [2.7, 1.9, 3.7, 0.8], 13, [], 0.36);
bullets(s, [
  "Final model: boosting + ridge, updated every lap (about 0.38 s per lap)",
  "Tire wear is predictable once a stint is underway; conditions alone are not enough",
  "Teams pit on strategy and information beyond public data; PitWall informs that decision",
], 0.45, 3.05, 9.1, 2.0, 15, 9);
s.addNotes(NOTES[11]);

// ================================================================ 13. Questions
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
s.addText("Questions", { placeholder: "title" });
s.addText("github.com/isaiahnick/f1-tire-degradation", { placeholder: "body" });
tires(s, 6.55, 3.75, 0.85, 0.2);
s.addNotes(NOTES[12]);

pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
});
