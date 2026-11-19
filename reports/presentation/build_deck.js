// PitWall final presentation (EM 538, Team 9). Run: node build.js
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/a69a8812-da28-4575-95a4-32b6eec2f8d8_4226a6e8-952f-4a4b-be1f-bea8c45734ee/pptx/scripts/apply_theme.js");

const FIG = "/home/claude/pw_laptop/reports/figures/slides/";
const EQ = "/home/claude/pptx_build/eq/";
const OUT = "/home/claude/pptx_build/PitWall_Final_Presentation.pptx";

const RED = "CC0000", INK = "000000", GREY = "595959", LIGHT = "F2F2F2", HEAD = "404040", HILITE = "FBE5E5";
const THEME = {
  name: "PitWall", headFontFace: "Arial", bodyFontFace: "Arial",
  colors: { dk1: "000000", lt1: "FFFFFF", dk2: "404040", lt2: "F2F2F2",
    accent1: "CC0000", accent2: "404040", accent3: "7F7F7F", accent4: "1F77B4",
    accent5: "FF7F0E", accent6: "2CA02C", hlink: "0563C1", folHlink: "954F72" },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "PitWall: Predicting F1 Tire Degradation to Recommend Pit Stop Timing";
pres.author = "Isaiah Nick, Anand Vadlamani";
pres.subject = "EM 538 Final Project Presentation";
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
const C = pres.SchemeColor;

const band = (h) => [
  { rect: { x: 0, y: 0, w: 10, h, fill: { color: RED }, line: { color: RED } } },
  { text: { text: "NC STATE UNIVERSITY",
    options: { x: 0.3, y: 0, w: 4, h, fontSize: 11, bold: true, color: "FFFFFF", valign: "middle", margin: 0 } } },
];
pres.defineSlideMaster({
  title: "TITLE", background: { color: "FFFFFF" },
  objects: [
    ...band(0.42),
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.0, w: 8.6, h: 1.3,
      fontSize: 30, bold: true, align: "center", valign: "middle", color: C.text1 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 2.45, w: 8.6, h: 1.3,
      fontSize: 16, align: "center", valign: "top", color: C.text2 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  slideNumber: { x: 9.2, y: 5.28, w: 0.5, h: 0.25, fontSize: 10, color: "7F7F7F", align: "right" },
  objects: [
    ...band(0.3),
    { text: { text: "EM 538 Final Project  |  Team 9", options: { x: 5.7, y: 0, w: 4.0, h: 0.3, fontSize: 10,
      color: "FFFFFF", align: "right", valign: "middle", margin: 0 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.38, w: 9.0, h: 0.62,
      fontSize: 26, bold: true, align: "center", valign: "middle", color: C.text1 }, text: "" } },
  ],
});

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
// equations are rendered at 250 dpi and 22 pt; `scale` sets the size on the slide
function eq(s, name, x, y, scale = 0.75) {
  const [pw, ph] = pngSize(EQ + name + ".png");
  const w = pw / 250 * scale, h = ph / 250 * scale;
  s.addImage({ path: EQ + name + ".png", x, y, w, h, altText: name + " equation" });
  return [w, h];
}
function bullets(s, items, x, y, w, h, size = 16) {
  const runs = items.map((t, i) => {
    const sub = Array.isArray(t);
    return { text: sub ? t[0] : t, options: { bullet: sub ? { indent: 18 } : true, indentLevel: sub ? 1 : 0,
      fontSize: sub ? size - 2 : size, breakLine: i < items.length - 1, paraSpaceAfter: 6 } };
  });
  s.addText(runs, { x, y, w, h, valign: "top", color: C.text1, isTextBox: true, margin: 2 });
}
function note(s, text, x, y, w, h, size = 13, opts = {}) {
  s.addText(text, { x, y, w, h, fontSize: size, color: GREY, valign: "top", isTextBox: true, margin: 2, ...opts });
}
function label(s, text, x, y, w) {
  s.addText(text, { x, y, w, h: 0.32, fontSize: 15, bold: true, color: RED, isTextBox: true, margin: 2 });
}
function table(s, rows, x, y, w, colW, size = 12, hilite = []) {
  const data = rows.map((r, i) => r.map((c, j) => ({ text: String(c), options: {
    bold: i === 0 || hilite.includes(i),
    fill: { color: i === 0 ? HEAD : (hilite.includes(i) ? HILITE : (i % 2 === 0 ? "F7F7F7" : "FFFFFF")) },
    color: i === 0 ? "FFFFFF" : INK, align: j === 0 ? "left" : "center", fontSize: size } })));
  s.addTable(data, { x, y, w, colW, border: { type: "solid", pt: 0.5, color: "BFBFBF" }, valign: "middle",
    rowH: 0.27, margin: 0.05 });
}
// tire sidewall: black rubber, compound-colored ring, grey rim
function tire(s, x, y, d, ring) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: "1A1A1A" }, line: { color: "1A1A1A" } });
  const r = d * 0.13;
  s.addShape(pres.shapes.OVAL, { x: x + r, y: y + r, w: d - 2 * r, h: d - 2 * r, fill: { color: "1A1A1A" },
    line: { color: ring, width: Math.max(2, d * 7) } });
  const q = d * 0.33;
  s.addShape(pres.shapes.OVAL, { x: x + q, y: y + q, w: d - 2 * q, h: d - 2 * q, fill: { color: "A6A6A6" },
    line: { color: "7F7F7F", width: 1 } });
}
function tires(s, cx, y, d) {
  const gap = 0.25, x0 = cx - (3 * d + 2 * gap) / 2;
  ["E10600", "FFD100", "F2F2F2"].forEach((c, i) => tire(s, x0 + i * (d + gap), y, d, c));
}
function box(s, x, y, w, h, head, body, emph = false) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: emph ? HILITE : LIGHT },
    line: { color: emph ? RED : "A6A6A6", width: 1 } });
  s.addText([{ text: head, options: { bold: true, fontSize: 13, color: INK, breakLine: true } },
    { text: body, options: { fontSize: 11, color: GREY } }],
  { x: x + 0.05, y: y + 0.05, w: w - 0.1, h: h - 0.1, valign: "top", align: "center", isTextBox: true, margin: 2 });
}
function arrow(s, x1, y, x2) {
  s.addShape(pres.shapes.LINE, { x: x1, y, w: x2 - x1, h: 0, line: { color: GREY, width: 1.5, endArrowType: "triangle" } });
}

// 1. Title -------------------------------------------------------------------
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
s.addText("PitWall: Predicting F1 Tire Degradation to Recommend Pit Stop Timing", { placeholder: "title" });
s.addText("Team 9: Isaiah Nick, Anand Vadlamani\nEM 538, Fall 2026\nInstructor: Dr. Fred Livingston", { placeholder: "body" });
tires(s, 5, 4.05, 0.8);
s.addNotes("Presenter: Anand. Introduce the team and the project.");

// 2. Problem -----------------------------------------------------------------
s = slideWith("Problem Definition and Significance", "Introduction");
bullets(s, [
  "Tire grip drops with use, so each lap on a set gets slower",
  "A pit stop costs about 20 s; stopping too early or too late loses race time",
  "Users: race strategists and tire engineers",
  "ML task: regression of lap time loss (s) for the remaining laps of a stint",
  "Benefit: an estimate of the pit window from data available during the race",
], 0.4, 1.15, 4.4, 3.9, 15);
const yb = img(s, "s_stint_example.png", 4.9, 1.15, 4.75);
note(s, "Dashed lines mark pit stops. Lap time rises within each stint and drops after new tires.", 4.9, yb + 0.05, 4.75, 0.5, 11);
s.addNotes("Presenter: Anand. A tire is a wearing part, like a cutting tool or a bearing: performance drops with use and the question is when to replace it. The plot is one 2025 race. Within each stint the lap times climb, and they drop again after each pit stop. A stop costs about 20 seconds, so the team has to trade that against the time lost on worn tires. We frame it as regression: predict how much slower each remaining lap of the stint will be.");

// 3. Objectives --------------------------------------------------------------
s = slideWith("Objectives", "Introduction");
table(s, [
  ["Objective", "Success criterion"],
  ["Predict lap time loss on a season not used for training (2025)", "Test MAE at least 15% below a linear wear baseline"],
  ["Recommend a pit lap from the predictions", "Within 3 laps of the team's actual stop for 60% of green-flag stops"],
  ["Dashboard for the pit wall", "Recommendation returned in under 1 s"],
], 0.6, 1.25, 8.8, [4.4, 4.4], 14);
note(s, "Both quantitative objectives come from our project proposal. Results are compared against them on the conclusions slide.", 0.6, 3.15, 8.8, 0.6, 13);
tires(s, 5, 3.95, 0.7);
s.addNotes("Presenter: Anand. These are the two quantitative objectives from our proposal. The first is prediction accuracy on an unseen season. The second is the decision: does the recommended lap agree with what the teams actually did. The third is the deliverable, a dashboard.");

// 4. System overview ---------------------------------------------------------
s = slideWith("System Overview", "Introduction");
const steps = [
  ["1. Ingest", "FastF1 archive, 92 races + practice sessions"],
  ["2. Clean", "10 filters, fuel correction"],
  ["3. Features", "Target, Setups A and B"],
  ["4. Train", "Grouped CV, grid search, LSTM"],
  ["5. Evaluate", "2025 test, error analysis"],
  ["6. Decide", "Pit window, backtest, dashboard"],
];
const bw = 1.3, gap = 0.28, x0 = (10 - (6 * bw + 5 * gap)) / 2;
steps.forEach(([h, b], i) => {
  const x = x0 + i * (bw + gap);
  box(s, x, 1.45, bw, 1.25, h, b, i === 5);
  if (i < 5) arrow(s, x + bw + 0.03, 2.07, x + bw + gap - 0.03);
});
bullets(s, [
  "Each stage is one command: python -m pitwall <stage>",
  "make all and make final rebuild every number in this presentation from the pinned data",
  "Python: pandas, scikit-learn, PyTorch, Streamlit, Plotly",
  "Inference for one stint: 0.03 to 0.35 s",
], 0.6, 3.1, 8.8, 2.0, 15);
s.addNotes("Presenter: Anand. This is the system end to end. Each box is one stage of the code. Ingest pulls the pinned data, cleaning keeps laps that reflect tire wear, features builds the target and the two feature setups, then training with grouped cross-validation, evaluation on 2025, and finally the pit window and the dashboard. Two make commands rebuild everything.");

// 5. Data --------------------------------------------------------------------
pres.addSection({ title: "Data" });
s = slideWith("Data Source and Preprocessing", "Data");
bullets(s, [
  "Source: FastF1 timing data, TracingInsights archive (pinned commits)",
  "92 races, 2022-2025, plus practice sessions",
  "Per lap: lap time, compound, tire age, stint, track status, position, gaps, weather",
  "Train: 2022-2024 (58 races, 42,102 laps)",
  "Test: 2025 (20 races, 15,711 laps), held out",
], 0.4, 1.15, 4.5, 3.9, 14);
table(s, [
  ["Cleaning step", "Laps remaining"],
  ["Raw laps", "101,297"],
  ["Remove wet races (12)", "88,676"],
  ["Remove Monaco", "86,014"],
  ["Remove lap 1, pit laps, SC/VSC, rain, timing flags", "72,759"],
  ["Remove outliers (107% rule, within-stint)", "69,615"],
  ["Target laps (stints with 6+ clean laps)", "57,813"],
], 5.1, 1.2, 4.5, [3.2, 1.3], 12, [6]);
s.addNotes("Presenter: Anand. The data comes from FastF1. Live timing now needs a paid login, so we use an archive that stores FastF1 output for every session, pinned to fixed commits so the data is reproducible. We only want laps where lap time reflects tire wear, so the table shows each cleaning step and how many laps are left. Training is 2022 to 2024 and 2025 is the test season.");

// 6. Target and EDA ----------------------------------------------------------
s = slideWith("Target Variable and Exploratory Analysis", "Data");
label(s, "Fuel correction", 0.4, 1.1, 4.3);
eq(s, "fuel", 0.45, 1.45, 0.6);
label(s, "Target (s)", 0.4, 1.95, 4.3);
eq(s, "target", 0.45, 2.32, 0.7);
bullets(s, [
  "Fuel effect estimated from our data: 0.031 s/kg",
  "The 3 reference laps are not used as targets",
  "Soft tires lose time fastest, then medium, then hard",
  "Late medium-tire laps come from few, managed stints (wide interval)",
], 0.4, 2.85, 4.3, 2.3, 13);
img(s, "s_degradation.png", 4.85, 1.15, 4.8);
s.addNotes("Presenter: Anand. Cars get faster as fuel burns, which hides tire wear, so we subtract a fuel effect: 110 kilograms burned linearly at 0.03 seconds per kilogram. Our own data estimates 0.031, so the assumption holds. The target is then how much slower a lap is than the median of the stint's first three clean laps. The plot is the training data by compound. Softs degrade fastest, which matches tire physics. The medium curve dips late because very few stints run that long.");

// 7. Features ----------------------------------------------------------------
pres.addSection({ title: "Methods" });
s = slideWith("Features", "Methods");
table(s, [
  ["Feature group", "Setup A", "Setup B"],
  ["Tire: compound, laps on set, tire age, age when fitted, new/used", "Yes", "Yes"],
  ["Race: laps remaining, race progress, stint number, position, gap ahead", "Yes", "Yes"],
  ["Circuit and team (one-hot)", "Yes", "Yes"],
  ["Practice long-run wear rate (s/lap, by race and compound)", "Yes", "Yes"],
  ["First 5 laps of the stint: mean loss, wear slope, laps ahead", "", "Yes"],
  ["First 5 laps of the stint: dirty-air share, mean gap ahead, car pace vs. field", "", "Yes"],
  ["Training laps", "42,102", "28,754"],
], 0.5, 1.15, 9.0, [6.6, 1.2, 1.2], 13);
label(s, "Setup A", 0.5, 3.72, 1.2);
note(s, "Information available before the stint starts (the framing in our proposal).", 1.6, 3.75, 7.9, 0.35, 13);
label(s, "Setup B", 0.5, 4.1, 1.2);
note(s, "Also uses the first 5 laps on the set; predicts the laps after them (the real pit-wall decision).", 1.6, 4.13, 7.9, 0.35, 13);
note(s, "Weather features were dropped: removing them lowered CV MAE for 5 of 8 tuned models, including the best one.", 0.5, 4.65, 9.0, 0.4, 12);
s.addNotes("Presenter: Isaiah. We use two feature setups. Setup A only uses what is known before a stint starts. Setup B adds what the pit wall has seen on the first five laps of the set, which is what a strategist actually has when deciding. The practice wear rate comes from Friday and Saturday long runs. Dirty air is the share of laps within one second of the car ahead. Car pace compares the stint's reference laps with the field median on the same laps.");

// 8. Methods -----------------------------------------------------------------
s = slideWith("Methods and Models", "Methods");
table(s, [
  ["Model", "Tuned hyperparameters (Setup B)"],
  ["Linear regression (baseline)", "loss = rate x laps on set, one rate per compound"],
  ["Ridge regression", "alpha = 1000"],
  ["Decision tree", "max depth 8, min leaf 100"],
  ["Random forest", "200 trees, min leaf 20, max features 0.6"],
  ["Gradient boosting (HistGB)", "learning rate 0.03, 300 iterations, 31 leaves"],
  ["GB + Ridge ensemble", "equal-weight average of the two tuned models"],
  ["LSTM (PyTorch)", "32 hidden units, dropout 0.2, learning rate 0.001"],
], 0.4, 1.12, 6.3, [2.4, 3.9], 12);
bullets(s, [
  "Train on 2022-2024, test on 2025",
  "5-fold GroupKFold by race",
  "Grid search inside the same folds",
  "Scaling and encoding inside scikit-learn Pipelines",
], 6.9, 1.12, 2.8, 2.6, 13);
const [w1] = eq(s, "mae", 0.6, 3.75, 0.6);
const [w2] = eq(s, "rmse", 0.6 + w1 + 0.6, 3.72, 0.6);
eq(s, "r2", 0.6 + w1 + w2 + 1.2, 3.68, 0.6);
note(s, "MAE is the primary metric: it is in seconds per lap, the unit a strategist uses.", 0.6, 4.6, 8.8, 0.4, 12);
s.addNotes("Presenter: Isaiah. The baseline is a straight-line wear rate per compound. The classical models are tuned with grid search, and the ensemble is a simple equal-weight average of gradient boosting and ridge. Cross-validation is grouped by race so laps from one race never appear in both training and validation; a random split would leak race conditions. Preprocessing sits inside Pipelines so it is only fit on training folds. MAE is our main metric because it is in seconds per lap.");

// 9. LSTM --------------------------------------------------------------------
s = slideWith("LSTM Model", "Methods");
const L = [["Input", "first 5 laps x 4 values"], ["LSTM", "32 hidden units"], ["Concatenate", "Setup A and B features"],
  ["Dense", "64, then 32"], ["Output", "loss for one lap (s)"]];
const lw = 1.55, lg = 0.3, lx0 = (10 - (5 * lw + 4 * lg)) / 2;
L.forEach(([h, b], i) => {
  const x = lx0 + i * (lw + lg);
  box(s, x, 1.15, lw, 0.75, h, b, i === 1);
  if (i < 4) arrow(s, x + lw + 0.03, 1.52, x + lw + lg - 0.03);
});
bullets(s, [
  "Sequence values: loss, laps on set, gap ahead, laps remaining",
  "L1 (MAE) loss, Adam optimizer",
  "Early stopping on 15% of training races; usually stops after about 3 epochs",
  "Same grouped folds as the other models",
], 0.4, 2.2, 4.6, 2.9, 13);
table(s, [
  ["Hidden units", "Dropout", "Learning rate", "CV MAE (s)"],
  ["16", "0.2", "0.001", "0.441"],
  ["32", "0.2", "0.001", "0.419"],
  ["64", "0.3", "0.001", "0.422"],
  ["64", "0.3", "0.0003", "0.424"],
  ["128", "0.4", "0.0003", "0.420"],
], 5.2, 2.2, 4.4, [1.2, 0.9, 1.2, 1.1], 12, [2]);
note(s, "Selected: 32 units. Re-run with 3 seeds: CV MAE 0.429 ± 0.009 s.", 5.2, 3.9, 4.4, 0.5, 12);
s.addNotes("Presenter: Isaiah. The LSTM reads the first five laps as a sequence, so it can learn the shape of the early wear instead of a summary of it. Its output is joined with the same static features the other models use, then two dense layers. We swept network size and learning rate on the same five folds and picked 32 units. Because neural network results depend on the random seed, we re-ran the chosen size with three seeds.");

// 10. Ablation ---------------------------------------------------------------
s = slideWith("Feature and Data Ablation", "Methods");
table(s, [
  ["Change (Setup B, 5-fold CV on 2023-2024)", "GB CV MAE (s)", "Ridge CV MAE (s)"],
  ["Original features, 2023-2024 data", "0.458", "0.463"],
  ["+ car pace vs. field", "0.455", "0.455"],
  ["+ dirty-air share and mean gap", "0.449", "0.456"],
  ["+ practice long-run wear", "0.452", "0.456"],
  ["+ all three", "0.440", "0.445"],
  ["+ 2022 races as extra training data", "0.442", "0.452"],
  ["Average of GB and Ridge (original features)", "0.451", "-"],
], 0.5, 1.15, 9.0, [5.4, 1.8, 1.8], 13, [5]);
note(s, "Tested with no gain (change under 0.2%): monotone constraints, MAE loss, smoothed training target.", 0.5, 3.72, 9.0, 0.35, 13);
note(s, "Final configuration (all changes kept), CV on 2022-2024: GB 0.428 s, Ridge 0.434 s, ensemble 0.421 s.", 0.5, 4.08, 9.0, 0.35, 13, { bold: true, color: INK });
note(s, "The 2025 test set was scored once before these changes (best: 0.421 s) and once after them.", 0.5, 4.45, 9.0, 0.35, 13);
s.addNotes("Presenter: Isaiah. After our first full run we tested changes one at a time on the same cross-validation folds, without using 2025. Each new feature helped a little and together they lowered error about 4 percent. Adding 2022 helped by about 3 percent and narrowed the gap between training and validation error. Some ideas did not help and were dropped. To be clear about the test set: we scored 2025 once before these changes and once after, and we report the second result.");

// 11. Evaluation -------------------------------------------------------------
pres.addSection({ title: "Results" });
s = slideWith("Model Evaluation", "Results");
table(s, [
  ["Setup B model", "CV MAE", "Train MAE", "Test MAE", "Test RMSE", "Test R²", "vs. baseline"],
  ["Linear baseline", "0.682", "0.667", "0.628", "0.797", "0.06", "-"],
  ["Decision tree", "0.488", "0.411", "0.483", "0.636", "0.40", "23%"],
  ["Random forest", "0.448", "0.277", "0.426", "0.566", "0.53", "32%"],
  ["Gradient boosting", "0.428", "0.284", "0.424", "0.557", "0.54", "32%"],
  ["Ridge", "0.434", "0.405", "0.419", "0.549", "0.55", "33%"],
  ["LSTM (mean of 3 seeds)", "0.429", "0.387", "0.422", "0.553", "0.54", "33%"],
  ["GB + Ridge ensemble", "0.421", "0.331", "0.412", "0.541", "0.57", "34%"],
], 0.4, 1.15, 9.2, [2.6, 1.0, 1.0, 1.0, 1.1, 1.0, 1.5], 13, [7]);
note(s, "All errors in seconds. Test = 2025 season. The ensemble was selected on CV MAE before testing.", 0.4, 3.7, 9.2, 0.35, 13);
s.addText([{ text: "34%", options: { fontSize: 30, bold: true, color: RED } },
  { text: "   lower test MAE than the linear baseline (objective: 15%)", options: { fontSize: 15, color: INK } }],
{ x: 0.4, y: 4.05, w: 9.2, h: 0.55, valign: "middle", isTextBox: true, margin: 2 });
note(s, "Setup A (conditions only): best test MAE 0.553 s (random forest), 5% below its baseline (0.580 s).", 0.4, 4.65, 9.2, 0.35, 13);
s.addNotes("Presenter: Isaiah. This is the main result on 2025. With the first five laps, every model beats the baseline by 23 to 34 percent, so the 15 percent objective is met. The ensemble was chosen on cross-validation before testing and is also best on test. The top five models are within about 0.015 seconds of each other, which is smaller than the fold-to-fold variation, so ridge is an equally reasonable, simpler choice. Setup A only reaches 5 percent: conditions alone do not tell you how a specific set of tires will wear.");

// 12. Over/underfitting ------------------------------------------------------
s = slideWith("Overfitting and Underfitting", "Results");
img(s, "s_train_test.png", 0.3, 1.08, 5.1);
img(s, "s_depth_curve.png", 5.7, 1.08, 3.9);
bullets(s, [
  "Linear baseline: train and test both high, R² 0.06 (underfitting)",
  "Random forest and boosting: train about 0.28 s vs. test 0.42-0.43 s (high variance)",
  "Ridge: train and test within 4%; averaging it with GB lowered test MAE to 0.412 s",
  "Single tree (right): validation error is lowest at depth 2 and rises with depth",
], 0.3, 4.02, 9.4, 1.15, 12);
s.addNotes("Presenter: Isaiah. We used the rule from class that training and test error should be within about 5 percent. The baseline is close but explains almost nothing, so it underfits. Random forest and boosting fit the training data much better than the test data, which is variance. Ridge is within 4 percent. Averaging boosting with ridge reduced the variance and gave the best test error. The depth curve shows the same thing for one tree: deeper trees memorize races.");

// 13. Error analysis ---------------------------------------------------------
s = slideWith("Error Analysis", "Results");
img(s, "s_horizon.png", 0.3, 1.08, 4.5);
img(s, "s_importance.png", 4.95, 1.08, 4.75);
note(s, "Highest 2025 error: Azerbaijan, Singapore, Canada (0.54-0.58 s). Lowest: São Paulo, Japan (0.28-0.29 s).\nBy compound (ensemble): soft 0.30 s, medium 0.37 s, hard 0.47 s.", 0.3, 4.4, 9.4, 0.7, 12);
s.addNotes("Presenter: Isaiah. Error grows with how far ahead we predict, from about 0.3 seconds within five laps to about 0.6 seconds past twenty. The most important feature is the mean loss over the first five laps, then how far ahead we predict, then the car's pace relative to the field. Street circuits had the highest error. Hards had the highest error because they run the longest stints.");

// 14. Forecast update --------------------------------------------------------
s = slideWith("Updating the Forecast During a Stint", "Results");
label(s, "Correction after observed lap d", 0.4, 1.1, 4.3);
eq(s, "update", 0.5, 1.47, 0.6);
bullets(s, [
  "Uses only completed laps; the model is not refit",
  "Number of errors averaged chosen by grouped CV (3, 5 or 8, with and without shrinkage)",
  "2025 MAE after lap 5: 0.500 s fixed vs. 0.379 s updated (24% lower)",
], 0.4, 2.2, 4.3, 2.9, 14);
img(s, "s_update.png", 4.85, 1.12, 4.8);
s.addNotes("Presenter: Isaiah. Setup B looks at the first five laps and then stops learning about the stint. In a race, the pit wall keeps seeing laps. So after each lap we check how far off the forecast was on the last three laps and shift the remaining forecast by that amount. Nothing is retrained and only completed laps are used, so there is no leakage. With the fixed forecast, error grows the longer the stint runs; with the update it stays near 0.38 seconds.");

// 15. Pit stop recommendation ------------------------------------------------
pres.addSection({ title: "Application" });
s = slideWith("Pit Stop Recommendation", "Application");
eq(s, "cost", 0.5, 1.12, 0.6);
note(s, "d = decision lap, E = end of the next stint, ŷB = Setup B (current set), ŷA = Setup A (next set), δ = compound pace offset. Pit lane time is the same for every p. Pit window: laps within 1 s of the minimum.", 0.5, 1.78, 9.0, 0.6, 12);
img(s, "s_pit_backtest.png", 0.3, 2.42, 5.0);
bullets(s, [
  "Backtest: 404 green-flag stops in 2025",
  "Learned layer: gradient boosting that predicts the stop lap from the cost-model window and race context (gap behind, position, stint, laps left), trained out-of-fold on 2022-2024",
  "It scores the same without the cost model: team timing depends mostly on race context",
  "60% objective not met",
], 5.5, 2.45, 4.2, 2.75, 12);
s.addNotes("Presenter: Isaiah. For every possible pit lap we add the predicted time lost on the current tires up to that lap and on the new tires after it, and pick the minimum. We compared the recommended lap with 404 real green-flag stops in 2025. The cost model with our best tire model is within three laps 25 percent of the time; the linear baseline does better at 39 percent because its curves are smoother. A learned layer trained on 2022 to 2024 reaches 55 percent, but it does just as well without the tire model, which tells us that when teams stop is driven by strategy such as undercuts and planned stop counts more than by the wear curve. We did not reach 60 percent.");

// 16. Demo -------------------------------------------------------------------
s = slideWith("Demonstration: Streamlit Dashboard", "Application");
const [dw, dh] = pngSize(FIG + "s_dashboard.png");
s.addShape(pres.shapes.RECTANGLE, { x: 0.28, y: 1.08, w: 6.64, h: 6.6 * dh / dw + 0.04, fill: { color: "FFFFFF" },
  line: { color: "A6A6A6", width: 1 } });
img(s, "s_dashboard.png", 0.3, 1.1, 6.6);
bullets(s, [
  "Select race, driver and stint (2025)",
  "Select the model for the current tires",
  "Move the decision lap to update the forecast",
  "Output: predicted curves, recommended lap, pit window, learned-layer lap, actual stop",
  "Second tab: loss curve per compound before a stint starts",
], 7.05, 1.12, 2.7, 3.9, 12);
s.addNotes("Presenter: Isaiah. Live demo, about two minutes. 1) In the terminal: python -m pitwall ingest pulls all races and practice sessions from the archive in about 15 seconds. 2) Dashboard default: 2025 Spanish GP, Verstappen, stint 2 on softs. Recommended lap 29, window 25 to 33, learned layer lap 30, actual stop lap 29. 3) Move the decision lap slider to show the forecast update. 4) Switch the model to the linear baseline or LSTM. 5) Plan a stint tab. Backup stints: Austria, Hamilton, stint 1 (25 vs 26); Saudi Arabia, Russell, stint 1 (19 vs 20). Open the app before class; the first load takes about 30 seconds.");

// 17. Limitations ------------------------------------------------------------
pres.addSection({ title: "Conclusion" });
s = slideWith("Limitations and Future Work", "Conclusion");
label(s, "Limitations", 0.4, 1.12, 4.5);
bullets(s, [
  "Public timing data only: no tire temperatures, fuel loads or team plans",
  "Fuel model assumes a full tank and a linear burn",
  "Backtest measures agreement with teams, not the best possible race time",
  "Needs 8 clean laps into a stint; wet races and Monaco excluded",
  "2025 test set evaluated twice (before and after the final changes)",
], 0.4, 1.5, 4.5, 3.6, 13);
label(s, "Future work", 5.1, 1.12, 4.5);
bullets(s, [
  "Test on 2026 cars (narrower tires, new regulations)",
  "Score pit decisions with a race-time simulation",
  "Calibrated prediction intervals (our quantile model covered 67% for an 80% interval)",
  "Add safety car probability and planned stop counts to the pit window",
], 5.1, 1.5, 4.5, 3.6, 13);
s.addNotes("Presenter: Isaiah. The main data limitation is that public timing has no tire temperatures or real fuel loads. The backtest compares us with what teams did, which is not necessarily optimal. We also looked at the 2025 test set twice, so the final number is slightly less clean than a single evaluation, even though every change was chosen on cross-validation. Next steps: 2026 cars, a race simulation to score pit calls by time, calibrated intervals, and safety car logic.");

// 18. Conclusions ------------------------------------------------------------
s = slideWith("Conclusions", "Conclusion");
table(s, [
  ["Objective", "Target", "Result (2025)", "Met"],
  ["Lap time loss prediction (Setup B)", "15% below baseline", "34% below (0.412 s vs. 0.628 s)", "Yes"],
  ["Pit lap within 3 laps of actual stop", "60% of stops", "25% cost model, 55% learned layer", "No"],
  ["Dashboard recommendation time", "under 1 s", "0.03 to 0.35 s", "Yes"],
], 0.5, 1.15, 9.0, [3.0, 1.7, 3.4, 0.9], 13);
bullets(s, [
  "Final model: GB + Ridge ensemble on the first 5 laps of the stint, corrected after each lap (24% lower error after lap 5)",
  "Tire wear is predictable once a stint has started; conditions alone are not enough (5% gain)",
  "When teams actually pit depends mainly on race strategy, so a pit tool needs strategy inputs as well as tire predictions",
], 0.5, 2.55, 9.0, 2.5, 15);
s.addNotes("Presenter: Isaiah. Against our objectives: the prediction target was met with room to spare, the dashboard is fast, and the pit timing objective was not met. Once a few laps on a set are known, the rest of the stint can be predicted about a third better than the baseline on an unseen season, and updating the forecast lap by lap helps further. The learned layer showed that team timing is mostly a strategy decision, so a better pit tool needs strategy information, not just better tire predictions.");

// 19. Questions --------------------------------------------------------------
s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Conclusion" });
s.addText("Questions", { placeholder: "title" });
s.addText("Code and data pipeline: github.com/isaiahnick/f1-tire-degradation", { placeholder: "body" });
tires(s, 5, 3.6, 0.8);

pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
});
