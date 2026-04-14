import JSZip from "jszip";
import type { FeatureSpec } from "@/lib/api";

function downloadText(
  name: string,
  text: string,
  mime = "text/plain;charset=utf-8"
) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function normalizeFeatureSpecForExport(featureSpec: FeatureSpec): Record<string, unknown> {
  const rangePattern =
    /(?:score|range|scale|hodnota)?\s*(\d+(?:\.\d+)?)\s*[-–—to]+\s*(\d+(?:\.\d+)?)/i;
  const binaryPattern = /\b(?:binary|bool|boolean)\b|(?:0\s+or\s+1)|(?:0\/1)/i;
  const percentagePattern = /\b(?:percent|percentage|%)\b/i;
  const enumHintPattern = /^(?:one of|enum|categories?|values?)\s*[:\-]?\s*/i;

  const normalized: Record<string, unknown> = {};

  for (const [key, raw] of Object.entries(featureSpec as Record<string, unknown>)) {
    if (Array.isArray(raw)) {
      if (raw.length === 2 && raw.every((v) => typeof v === "number")) {
        const lo = Number(raw[0]);
        const hi = Number(raw[1]);
        normalized[key] = lo <= hi ? [lo, hi] : [hi, lo];
        continue;
      }

      if (raw.every((v) => typeof v === "string")) {
        const vals = raw.map((v) => String(v).trim()).filter(Boolean);
        if (vals.length > 0) normalized[key] = vals;
        continue;
      }
    }

    if (typeof raw !== "string") {
      normalized[key] = raw;
      continue;
    }

    const text = raw.trim();
    if (!text) continue;

    if (binaryPattern.test(text)) {
      normalized[key] = [0, 1];
      continue;
    }

    if (percentagePattern.test(text)) {
      normalized[key] = [0, 100];
      continue;
    }

    const match = text.match(rangePattern);
    if (match) {
      const lo = Number(match[1]);
      const hi = Number(match[2]);
      normalized[key] = lo <= hi ? [lo, hi] : [hi, lo];
      continue;
    }

    const enumText = text
      .replace(enumHintPattern, "")
      .trim()
      .replace(/^[\[\(\{]\s*/, "")
      .replace(/\s*[\]\)\}]$/, "");
    const splitBy = enumText.includes(",") ? "," : (enumText.includes("|") ? "|" : null);
    if (splitBy) {
      const values = enumText
        .split(splitBy)
        .map((value) => value.trim().replace(/^['"]|['"]$/g, ""))
        .filter(Boolean);
      if (values.length >= 2) {
        normalized[key] = values;
        continue;
      }
    }

    normalized[key] = text;
  }

  return normalized;
}

function rowsToCsv(rows: Record<string, unknown>[]) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  return [
    headers.join(","),
    ...rows.map((row) =>
      headers
        .map((header) => {
          const value = row[header];
          if (value === null || value === undefined) return "";
          const text =
            value !== null && typeof value === "object"
              ? JSON.stringify(value)
              : String(value);
          if (text.includes(",") || text.includes('"') || text.includes("\n")) {
            return `"${text.replace(/"/g, '""')}"`;
          }
          return text;
        })
        .join(",")
    ),
  ].join("\n");
}

export function downloadFeatureSpec(
  featureSpec: FeatureSpec,
  filename = `feature_spec_${new Date().toISOString().slice(0, 10)}.json`
) {
  const normalized = normalizeFeatureSpecForExport(featureSpec);
  const json = JSON.stringify(normalized, null, 2);
  downloadText(filename, json, "application/json;charset=utf-8");
}

export function downloadTrainingDataCSV(
  trainingData: Record<string, unknown>[],
  filename = `training_dataset_X_${new Date().toISOString().slice(0, 10)}.csv`
) {
  if (!trainingData || trainingData.length === 0) return;
  downloadText(filename, rowsToCsv(trainingData), "text/csv;charset=utf-8");
}

export function downloadRulesModel(
  rules: string[],
  mse?: number,
  filename = `rules_model_${new Date().toISOString().slice(0, 10)}.json`
) {
  const model = {
    rules,
    mse,
    count: rules.length,
    timestamp: new Date().toISOString(),
  };
  downloadText(filename, JSON.stringify(model, null, 2), "application/json;charset=utf-8");
}

export function downloadTestingDataWithPrediction(
  testingData: Record<string, unknown>,
  prediction: number,
  ruleApplied: string,
  filename = `testing_prediction_${new Date().toISOString().slice(0, 10)}.json`
) {
  const result = {
    testing_data_X: testingData,
    prediction_score: prediction,
    rule_applied: ruleApplied,
    timestamp: new Date().toISOString(),
  };
  downloadText(filename, JSON.stringify(result, null, 2), "application/json;charset=utf-8");
}

export async function downloadExperimentZip(params: {
  featureSpec?: FeatureSpec | null;
  trainingDataX?: Record<string, unknown>[] | null;
  testingDataX?: Record<string, unknown>[] | null;
  rules?: string[] | null;
  mse?: number;
  predictions?: Record<string, unknown>[] | null;
  metrics?: Record<string, unknown> | null;
}) {
  const zip = new JSZip();
  const ts = new Date().toISOString().slice(0, 19).replace(/:/g, "-");

  if (params.featureSpec) {
    zip.file("feature_spec.json", JSON.stringify(normalizeFeatureSpecForExport(params.featureSpec), null, 2));
  }
  if (params.trainingDataX && params.trainingDataX.length > 0) {
    zip.file("training_dataset_X.csv", rowsToCsv(params.trainingDataX));
  }
  if (params.testingDataX && params.testingDataX.length > 0) {
    zip.file("testing_dataset_X.csv", rowsToCsv(params.testingDataX));
  }
  if (params.rules) {
    zip.file(
      "rules_model.json",
      JSON.stringify(
        {
          rules: params.rules,
          mse: params.mse,
          count: params.rules.length,
          timestamp: new Date().toISOString(),
        },
        null,
        2
      )
    );
  }
  if (params.predictions && params.predictions.length > 0) {
    zip.file("predictions.csv", rowsToCsv(params.predictions));
    if (params.metrics) {
      zip.file("metrics.json", JSON.stringify(params.metrics, null, 2));
    }
  }

  const blob = await zip.generateAsync({ type: "blob" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `experiment_${ts}.zip`;
  a.click();
  URL.revokeObjectURL(url);
}
