import type { FeatureSpec, LlmEndpointConfig, TargetMode } from "@/lib/api";

export interface PersistedPipeline {
  trainingStep?: 1 | 2 | 3 | 4 | 5;
  targetVariable?: string;
  targetMode?: TargetMode;
  featureSpec?: FeatureSpec | null;
  modelProvider?: string;
  llmEndpoint?: LlmEndpointConfig;
}

const STORAGE_KEY = "mflPipeline";
const ACTIVE_JOB_KEY = "mflActiveJob";

export type ActiveJobPhase =
  | "discover"
  | "extract_training"
  | "extract_testing"
  | "train"
  | "predict";

export interface ActiveJob {
  job_id: string;
  phase: ActiveJobPhase;
}

export function loadPersisted(): PersistedPipeline | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function savePersisted(data: PersistedPipeline) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {
    /* quota exceeded or localStorage unavailable */
  }
}

export function clearPersisted() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* localStorage unavailable */
  }
}

export function saveActiveJob(job: ActiveJob) {
  try {
    localStorage.setItem(ACTIVE_JOB_KEY, JSON.stringify(job));
  } catch {
    /* localStorage unavailable */
  }
}

export function clearActiveJob() {
  try {
    localStorage.removeItem(ACTIVE_JOB_KEY);
  } catch {
    /* localStorage unavailable */
  }
}

export function loadActiveJob(): ActiveJob | null {
  try {
    const raw = localStorage.getItem(ACTIVE_JOB_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

const TRAINING_PIPELINE_TEXT = {
  en: {
    restoring: "Restoring...",
    phase1Failed: "Phase 1 failed",
    phaseFailed: "failed",
    extractFailed: "Extraction failed",
    featureDiscoveryFailed: "Feature Discovery failed",
    startingExtraction: "Starting extraction...",
    phase2: "Phase 2",
    phase4: "Phase 4",
    startingTraining: "Starting training...",
    trainingFailed: "Training failed",
    trainingInProgress: "Training...",
    phase3Failed: "Phase 3 failed",
    startingPrediction: "Starting prediction...",
    predictionFailed: "Prediction failed",
    predictingInProgress: "Predicting...",
    phase5Failed: "Phase 5 failed",
    reconnecting: "Reconnecting to server…",
    connectionLost: "connection to server was lost",
  },
  cs: {
    restoring: "Obnovuji...",
    phase1Failed: "Fáze 1 selhala",
    phaseFailed: "selhala",
    extractFailed: "Extrakce selhala",
    featureDiscoveryFailed: "Objevování featur selhalo",
    startingExtraction: "Spouštím extrakci...",
    phase2: "Fáze 2",
    phase4: "Fáze 4",
    startingTraining: "Spouštím trénink...",
    trainingFailed: "Trénink selhal",
    trainingInProgress: "Trénuji...",
    phase3Failed: "Fáze 3 selhala",
    startingPrediction: "Spouštím predikci...",
    predictionFailed: "Predikce selhala",
    predictingInProgress: "Predikuji...",
    phase5Failed: "Fáze 5 selhala",
    reconnecting: "Obnovuji spojení se serverem…",
    connectionLost: "spojení se serverem bylo přerušeno",
  },
} as const;

export function getTrainingPipelineText(uiLanguage: "cs" | "en" = "cs") {
  return TRAINING_PIPELINE_TEXT[uiLanguage];
}
