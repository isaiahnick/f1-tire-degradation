// Speaker notes, one entry per slide in order. Timing adds to about 8 min plus the demo.
const N = (...lines) => lines.join("\n");
module.exports = [
  // 1 Title
  N("ANAND, 15 s",
    "- We're Team 9. PitWall predicts how fast F1 tires wear during a race and uses it to recommend when to pit."),
  // 2 Problem and objectives
  N("ANAND, 50 s",
    "- Tires wear like a cutting tool: grip drops every lap, so lap time rises.",
    "- In this race, lap time climbs through each stint and drops after each stop.",
    "- A stop costs about 20 s, so the question is when worn tires cost more than stopping.",
    "- We predict how many seconds slower each remaining lap will be, then recommend a pit lap.",
    "- Two goals: beat a linear baseline by 15% on 2025, and land within 3 laps of the team's real stop 60% of the time, excluding safety car stops."),
  // 3 Data
  N("ANAND, 50 s",
    "- Data is FastF1, the official F1 timing, from a saved archive so the data is reproducible.",
    "- We keep only laps where lap time reflects tire wear: no rain, no Monaco, no lap 1, pit or safety car laps, no outliers.",
    "- Cars start with up to 110 kg of fuel and get about 0.03 s faster per kg burned, which hides tire wear, so we subtract it. Our data estimated 0.031.",
    "- Target: seconds slower than the stint's first 3 laps, the time lost to tire wear.",
    "- Softs wear fastest, then mediums, then hards. The late medium dip is from a few long, managed stints.",
    "- We train on 2022 to 2024 and test on 2025, which is never used for training or tuning."),
  // 4 Features
  N("ANAND, 50 s",
    "- Laps 1 to 3 set the reference, laps 4 to 8 are what the pit wall has seen, and we predict the rest.",
    "- Setup A uses only what's known before the stint. Setup B adds the first 5 laps, the real pit-wall situation.",
    "- New features: practice long-run wear, dirty air, since running close behind a car overheats tires, and car pace vs. the field.",
    "- We kept a change only if cross-validation improved. Weather didn't, so it's out.",
    "- Isaiah will cover the models and results."),
  // 5 Models and validation
  N("ISAIAH, 50 s",
    "- The baseline is linear regression with one wear rate per compound.",
    "- We compared simple to complex: ridge, a decision tree, random forest, gradient boosting, and an LSTM, a neural network that reads the laps as a sequence.",
    "- Cross-validation is grouped by race, so a race is never split between training and validation. A random split would leak race conditions and inflate scores.",
    "- Hyperparameters come from grid search on the same folds; the LSTM worked best at 32 units.",
    "- Our final model averages gradient boosting and ridge. It had the lowest CV error, since a flexible and a smooth model partly cancel each other's errors."),
  // 6 Evaluation
  N("ISAIAH, 60 s",
    "- These are results on 2025, which the models never saw.",
    "- MAE is the average miss per lap: 0.41 s for our model, 0.63 s for the baseline. That's 34% better, against a 15% goal.",
    "- For scale, a soft tire is about 1 s slower after 20 laps.",
    "- Light bars are training error, dark bars test. Random forest and boosting overfit; the baseline underfits; ridge is close on both.",
    "- The top models are within a hundredth of a second, so ridge alone is an equally good, simpler choice.",
    "- With conditions only, we're just 5% better: you have to see how this set of tires runs."),
  // 7 Works and fails
  N("ISAIAH, 40 s",
    "- The biggest driver is how much time the tires lost in the first 5 laps, then how far ahead we predict, then car pace.",
    "- A forecast made at lap 5 gets worse the further ahead it looks: 0.31 s per lap within 5 laps, 0.60 past 20.",
    "- It struggles most on street circuits and on hard tires, which run the longest stints.",
    "- Because a frozen forecast drifts, our final model keeps updating itself."),
  // 8 Forecast update
  N("ISAIAH, 45 s",
    "- After every lap, we check how far off the forecast was over the last 3 laps and shift the rest of the forecast by that amount.",
    "- It only uses laps that have happened, and nothing is retrained.",
    "- The frozen forecast in grey climbs to 0.63; the updated one in red stays near 0.38, 24% lower.",
    "- This is what the decision-lap slider does in the dashboard."),
  // 9 Pit decisions
  N("ISAIAH, 60 s",
    "- These are two different questions: when is it cheapest to stop, and when do teams actually stop.",
    "- The tire model adds up predicted time lost on old and new tires for every possible pit lap; the lowest total is the tire-optimal window.",
    "- Against 404 real 2025 stops, it's within 3 laps 25% of the time.",
    "- A second model trained on 1,230 real stops from race context gets 56%, and adding our tire predictions doesn't change it.",
    "- So teams decide on strategy and information that isn't public, like planned stop counts, undercuts and tire allocation.",
    "- PitWall supports that call: the tire-optimal window next to what teams typically do. We didn't reach 60%, and matching teams may not be the right test of a good pit call."),
  // 10 Demo
  N("ISAIAH, 2 to 4 min",
    "1. Terminal: python -m pitwall ingest pulls every race and practice session in about 15 s.",
    "2. Dashboard: Spanish GP, Verstappen, stint 2. Dots are laps seen, the blue line is the forecast, the dashed line is new mediums, shading is the pit window.",
    "3. Recommended lap 29, typical team call 30, actual stop 29. The blue box explains the call and the inference time.",
    "4. Drag the decision-lap slider: the forecast updates.",
    "5. Switch to the linear baseline or LSTM: the call moves.",
    "6. Plan a stint tab: the loss curve for each compound before a stint starts.",
    "Backups: Austria, Hamilton, stint 1 (25 vs. 26). Saudi Arabia, Russell, stint 1 (19 vs. 20)."),
  // 11 Limitations
  N("ISAIAH, 30 s",
    "- Public timing has no tire temperatures, fuel loads or team plans, which limits both predictions and pit matching.",
    "- The fuel effect is an assumption, though our data agreed with it.",
    "- The backtest measures agreement with teams, not the fastest possible race.",
    "- We scored 2025 twice, before and after our final improvements, with every change chosen by cross-validation first.",
    "- Next: 2026 cars with new, narrower tires, a race simulation to score pit calls, calibrated intervals and safety car risk."),
  // 12 Conclusions
  N("ISAIAH, 30 s",
    "- Our prediction goal was met at 34% against 15%, the dashboard is fast, and the pit-timing goal was not met.",
    "- Tire wear is predictable once a stint is underway, especially when the forecast updates every lap.",
    "- When to pit is a strategy decision; PitWall informs it rather than replacing it.",
    "- Thank you, we're happy to take questions."),
  // 13 Questions
  N("BOTH"),
];
