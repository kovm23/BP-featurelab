import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import type {
  FeatureSpec,
  PipelineState,
  PredictionItem,
  PredictionMetrics,
  TargetMode,
} from "@/lib/api";
import { savePersisted } from "@/hooks/trainingPipelineUtils";

type Step = 1 | 2 | 3 | 4 | 5;
type Setter<T> = Dispatch<SetStateAction<T>>;

export interface PipelineHydrationSetters {
  setFeatureSpec: Setter<FeatureSpec | null>;
  setTargetVariable: Setter<string>;
  setTargetMode: Setter<TargetMode>;
  setTrainingDataX: Setter<Record<string, unknown>[] | null>;
  setDatasetYColumns: Setter<string[] | null>;
  setTrainResult: Setter<PipelineState["train_result"]>;
  setTestingDataX: Setter<Record<string, unknown>[] | null>;
  setPredictions: Setter<PredictionItem[] | null>;
  setPredictionMetrics: Setter<PredictionMetrics | null>;
  setTrainingStep: Setter<Step>;
}

export function applyBackendPipelineState(
  state: PipelineState,
  setters: PipelineHydrationSetters,
) {
  if (!state.completed_phases || state.completed_phases.length === 0) return;

  if (state.feature_spec && Object.keys(state.feature_spec).length > 0) {
    setters.setFeatureSpec(state.feature_spec);
  }
  if (state.target_variable) {
    setters.setTargetVariable(state.target_variable);
  }
  if (state.target_mode) {
    setters.setTargetMode(state.target_mode);
  }
  if (state.training_data_X && state.training_data_X.length > 0) {
    setters.setTrainingDataX(state.training_data_X);
  }
  if (state.dataset_Y_columns) {
    setters.setDatasetYColumns(state.dataset_Y_columns);
  }
  if (state.train_result) {
    setters.setTrainResult(state.train_result);
  }
  if (state.testing_data_X && state.testing_data_X.length > 0) {
    setters.setTestingDataX(state.testing_data_X);
  }
  if (state.predictions && state.predictions.length > 0) {
    setters.setPredictions(state.predictions);
  }
  if (state.prediction_metrics) {
    setters.setPredictionMetrics(state.prediction_metrics);
  }
  setters.setTrainingStep((prev) => {
    const suggested = Math.min(state.suggested_step as Step, 5) as Step;
    return suggested > prev ? suggested : prev;
  });
}

export function persistDiscoveryOutcome(params: {
  featureSpec: FeatureSpec;
  targetVariableRef: MutableRefObject<string>;
  targetModeRef: MutableRefObject<TargetMode>;
  modelProviderRef: MutableRefObject<string>;
}) {
  savePersisted({
    trainingStep: 2,
    targetVariable: params.targetVariableRef.current,
    targetMode: params.targetModeRef.current,
    featureSpec: params.featureSpec,
    modelProvider: params.modelProviderRef.current,
  });
}
