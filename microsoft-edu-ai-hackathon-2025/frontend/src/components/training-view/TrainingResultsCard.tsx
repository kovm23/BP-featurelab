import { CheckCircle2, ChevronRight, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { TrainResult } from "@/lib/api";
import {
  downloadFeatureSpec,
  downloadRulesModel,
  downloadTrainingDataCSV,
} from "@/lib/helpers";
import { cls } from "./shared";
import type { TrainingTranslations } from "./translations";

export function TrainingResultsCard({
  deluxe,
  trainResult,
  tr,
  onGoToStep,
}: {
  deluxe: boolean;
  trainResult: TrainResult;
  tr: TrainingTranslations;
  onGoToStep?: (step: number) => void;
}) {
  if (trainResult.status !== "success") return null;

  const featureImportance =
    trainResult.feature_importance?.xgboost &&
    Object.keys(trainResult.feature_importance.xgboost).length > 0
      ? trainResult.feature_importance.xgboost
      : (trainResult.feature_importance?.rulekit ?? {});

  return (
    <div
      className={`rounded-xl p-4 space-y-3 border ${cls(
        deluxe,
        "bg-green-50 border-green-100",
        "bg-green-900/20 border-green-800/50"
      )}`}
    >
      <div className={`flex items-center gap-2 ${cls(deluxe, "text-green-800", "text-green-400")}`}>
        <CheckCircle2 className="h-4 w-4" />
        <p className="text-sm font-bold">{tr.trainingDone}</p>
      </div>

      <div className={`text-xs space-y-0.5 ${cls(deluxe, "text-green-700", "text-green-400/80")}`}>
        {trainResult.target_mode === "classification" ? (
          <>
            <p>
              Train Accuracy:{" "}
              <strong>
                {trainResult.train_accuracy != null ? Number(trainResult.train_accuracy).toFixed(4) : "—"}
              </strong>
            </p>
            {trainResult.train_balanced_accuracy != null && (
              <p>
                {tr.balancedAccuracy}:{" "}
                <strong>{Number(trainResult.train_balanced_accuracy).toFixed(4)}</strong>
              </p>
            )}
            <p>
              Train F1 macro:{" "}
              <strong>
                {trainResult.train_f1_macro != null ? Number(trainResult.train_f1_macro).toFixed(4) : "—"}
              </strong>
            </p>
            {trainResult.train_mcc != null && (
              <p>
                {tr.matthews}: <strong>{Number(trainResult.train_mcc).toFixed(4)}</strong>
              </p>
            )}
            {trainResult.cv_accuracy != null && (
              <p>
                Cross-val Accuracy ({trainResult.cv_folds ?? 5}-fold):{" "}
                <strong>{Number(trainResult.cv_accuracy).toFixed(4)}</strong>
              </p>
            )}
            {trainResult.cv_balanced_accuracy != null && (
              <p>
                {tr.balancedAccuracy} CV:{" "}
                <strong>{Number(trainResult.cv_balanced_accuracy).toFixed(4)}</strong>
              </p>
            )}
            {trainResult.cv_f1_macro != null && (
              <p>
                Cross-val F1 macro: <strong>{Number(trainResult.cv_f1_macro).toFixed(4)}</strong>
              </p>
            )}
            {trainResult.cv_precision_macro != null && (
              <p>
                Cross-val Precision:{" "}
                <strong>{Number(trainResult.cv_precision_macro).toFixed(4)}</strong>
              </p>
            )}
            {trainResult.cv_recall_macro != null && (
              <p>
                Cross-val Recall: <strong>{Number(trainResult.cv_recall_macro).toFixed(4)}</strong>
              </p>
            )}
            {trainResult.cv_mcc != null && (
              <p>
                {tr.matthews} CV: <strong>{Number(trainResult.cv_mcc).toFixed(4)}</strong>
              </p>
            )}
          </>
        ) : (
          <>
            <p>
              Ensemble MSE: <strong>{trainResult.mse != null ? Number(trainResult.mse).toFixed(4) : "—"}</strong>
            </p>
            {trainResult.rulekit_mse != null && (
              <p>RuleKit MSE: <strong>{Number(trainResult.rulekit_mse).toFixed(4)}</strong></p>
            )}
            {trainResult.xgb_mse != null && (
              <p>XGBoost MSE: <strong>{Number(trainResult.xgb_mse).toFixed(4)}</strong></p>
            )}
            {trainResult.cv_mse != null && (
              <p>
                Cross-val MSE ({trainResult.cv_folds ?? 5}-fold):{" "}
                <strong>{Number(trainResult.cv_mse).toFixed(4)}</strong>
                {trainResult.cv_std != null && (
                  <span className="opacity-70"> ± {Number(trainResult.cv_std).toFixed(4)}</span>
                )}{" "}
                <span
                  className={
                    trainResult.cv_mse < (trainResult.mse ?? 0) * 2 ? "text-green-500" : "text-amber-500"
                  }
                >
                  {trainResult.cv_mse < (trainResult.mse ?? 0) * 2 ? "✓ model zobecňuje" : "⚠ možné přetrénování"}
                </span>
              </p>
            )}
            {trainResult.cv_mae != null && (
              <p>Cross-val MAE: <strong>{Number(trainResult.cv_mae).toFixed(4)}</strong></p>
            )}
          </>
        )}
        {trainResult.warnings?.map((warning, index) => (
          <p key={index} className="text-amber-600">
            ⚠ {warning}
          </p>
        ))}
        <p>
          {tr.rulesGenerated}: <strong>{trainResult.rules_count}</strong>
        </p>
      </div>

      {Object.keys(featureImportance).length > 0 && (
        <details className={`text-xs rounded border p-2 ${cls(deluxe, "border-slate-200", "border-slate-700")}`}>
          <summary className={`cursor-pointer font-medium ${cls(deluxe, "text-slate-700", "text-slate-300")}`}>
            {tr.featureImportanceTop}
          </summary>
          <ul className="mt-1.5 space-y-1">
            {Object.entries(featureImportance)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 5)
              .map(([feature, score]) => (
                <li key={feature} className={`flex items-center gap-2 ${cls(deluxe, "text-slate-600", "text-slate-400")}`}>
                  <span className="font-mono flex-1">{feature}</span>
                  <div
                    className="h-1.5 rounded-full bg-blue-400"
                    style={{ width: `${Math.round(score * 100)}px`, minWidth: "4px", maxWidth: "120px" }}
                  />
                  <span className="w-10 text-right">{(score * 100).toFixed(1)}%</span>
                </li>
              ))}
          </ul>
        </details>
      )}

      {trainResult.rules && trainResult.rules.length > 0 && (
        <div
          className={`p-3 rounded border text-left ${cls(
            deluxe,
            "bg-slate-100 border-slate-300",
            "bg-slate-800 border-slate-700"
          )}`}
        >
          <p className="font-bold mb-2 text-xs">{tr.rulesModelTitle}</p>
          <div className={`max-h-40 overflow-y-auto text-xs font-mono space-y-1 ${cls(deluxe, "text-slate-600", "text-slate-300")}`}>
            {trainResult.rules.map((rule, idx) => (
              <div key={idx} className="whitespace-pre-wrap break-words">
                {idx + 1}. {rule}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2 mt-3">
        <button
          onClick={() => {
            if (trainResult.feature_spec) downloadFeatureSpec(trainResult.feature_spec);
          }}
          className="px-2 py-1 bg-zinc-600 text-white rounded text-xs hover:bg-zinc-700 flex items-center justify-center gap-1"
        >
          <Download className="w-3 h-3" /> Feature Spec
        </button>
        <button
          onClick={() => {
            if (trainResult.training_data_X) downloadTrainingDataCSV(trainResult.training_data_X);
          }}
          className="px-2 py-1 bg-zinc-600 text-white rounded text-xs hover:bg-zinc-700 flex items-center justify-center gap-1"
        >
          <Download className="w-3 h-3" /> Training Data (X)
        </button>
        <button
          onClick={() => {
            if (trainResult.rules) downloadRulesModel(trainResult.rules, trainResult.mse);
          }}
          className="col-span-2 px-2 py-1 bg-zinc-600 text-white rounded text-xs hover:bg-zinc-700 flex items-center justify-center gap-1"
        >
          <Download className="w-3 h-3" /> {tr.downloadRuleModel}
        </button>
      </div>

      {onGoToStep && (
        <div className="flex justify-center mt-4">
          <Button onClick={() => onGoToStep(4)}>
            <ChevronRight className="mr-2 h-4 w-4" /> {tr.continue4}
          </Button>
        </div>
      )}
    </div>
  );
}
