import type { Dispatch, SetStateAction } from "react";
import type {
  FeatureSpec,
  PredictionItem,
  PredictionMetrics,
  TargetMode,
  TrainResult,
} from "@/lib/api";

type Step = 1 | 2 | 3 | 4 | 5;
type Setter<T> = Dispatch<SetStateAction<T>>;

export interface PipelineStateSetters {
  setFeatureSpec: Setter<FeatureSpec | null>;
  setTrainingDataX: Setter<Record<string, unknown>[] | null>;
  setDatasetYColumns: Setter<string[] | null>;
  setTrainResult: Setter<TrainResult | null>;
  setTestingDataX: Setter<Record<string, unknown>[] | null>;
  setPredictions: Setter<PredictionItem[] | null>;
  setPredictionMetrics: Setter<PredictionMetrics | null>;
  setTrainingStep: Setter<Step>;
}

export interface PipelineRuntimeSetters {
  setIsDiscovering: Setter<boolean>;
  setExtractionBusy: Setter<boolean>;
  setTrainingBusy: Setter<boolean>;
  setTestExtractionBusy: Setter<boolean>;
  setPredictBusy: Setter<boolean>;
  setProgress: Setter<number>;
  setProgressLabel: Setter<string>;
  setError: Setter<string | null>;
  setTargetVariable: Setter<string>;
  setTargetMode: Setter<TargetMode>;
  setFeatureSpec: Setter<FeatureSpec | null>;
  setTrainingDataX: Setter<Record<string, unknown>[] | null>;
  setDatasetYColumns: Setter<string[] | null>;
  setTrainResult: Setter<TrainResult | null>;
  setTestingDataX: Setter<Record<string, unknown>[] | null>;
  setPredictions: Setter<PredictionItem[] | null>;
  setPredictionMetrics: Setter<PredictionMetrics | null>;
  setTrainingStep: Setter<Step>;
}

export function invalidatePipelineFromPhase(setters: PipelineStateSetters, startPhase: Step) {
  if (startPhase <= 1) setters.setFeatureSpec(null);
  if (startPhase <= 2) {
    setters.setTrainingDataX(null);
    setters.setDatasetYColumns(null);
  }
  if (startPhase <= 3) setters.setTrainResult(null);
  if (startPhase <= 4) setters.setTestingDataX(null);
  if (startPhase <= 5) {
    setters.setPredictions(null);
    setters.setPredictionMetrics(null);
  }
  setters.setTrainingStep((prev) => (prev > startPhase ? startPhase : prev));
}

export function clearActivePipelineRuntime(setters: Pick<
  PipelineRuntimeSetters,
  | "setExtractionBusy"
  | "setTrainingBusy"
  | "setTestExtractionBusy"
  | "setPredictBusy"
  | "setIsDiscovering"
  | "setProgress"
  | "setProgressLabel"
>) {
  setters.setExtractionBusy(false);
  setters.setTrainingBusy(false);
  setters.setTestExtractionBusy(false);
  setters.setPredictBusy(false);
  setters.setIsDiscovering(false);
  setters.setProgress(0);
  setters.setProgressLabel("");
}

export function resetPipelineLocalState(
  setters: PipelineRuntimeSetters,
  defaults = {
    targetVariable: "movie memorability score",
    targetMode: "regression" as TargetMode,
  }
) {
  setters.setTrainingStep(1);
  setters.setTargetVariable(defaults.targetVariable);
  setters.setTargetMode(defaults.targetMode);
  setters.setFeatureSpec(null);
  setters.setIsDiscovering(false);
  setters.setExtractionBusy(false);
  setters.setTrainingDataX(null);
  setters.setDatasetYColumns(null);
  setters.setTrainingBusy(false);
  setters.setTrainResult(null);
  setters.setTestExtractionBusy(false);
  setters.setTestingDataX(null);
  setters.setPredictBusy(false);
  setters.setPredictions(null);
  setters.setPredictionMetrics(null);
  setters.setProgress(0);
  setters.setProgressLabel("");
  setters.setError(null);
}
