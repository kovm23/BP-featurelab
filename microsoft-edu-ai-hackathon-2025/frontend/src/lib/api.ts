// =====================================================
// API URL KONFIGURACE
// =====================================================
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export const DISCOVER_URL = `${API_BASE}/discover`;
export const EXTRACT_URL = `${API_BASE}/extract`;
export const TRAIN_URL = `${API_BASE}/train`;
export const PREDICT_URL = `${API_BASE}/predict`;
export const EXTRACT_LOCAL_URL = `${API_BASE}/extract-local`;
export const RESET_URL = `${API_BASE}/reset`;
export const STATE_URL = `${API_BASE}/state`;
export const HEALTH_URL = `${API_BASE}/health`;
export const QUEUE_INFO_URL = `${API_BASE}/queue-info`;
export const EXPORT_SESSION_URL = `${API_BASE}/export-session`;
export const IMPORT_SESSION_URL = `${API_BASE}/import-session`;
export const STATUS_URL = (jobId: string) =>
  `${API_BASE}/status/${encodeURIComponent(jobId)}`;

// =====================================================
// SESSION — perzistentní X-Session-ID header
// =====================================================

const _SESSION_KEY = "ml_session_id";

function getSessionId(): string {
  let id = localStorage.getItem(_SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(_SESSION_KEY, id);
  }
  return id;
}

/** Returns the X-Session-ID header object to spread into fetch options. */
export function sessionHeaders(): Record<string, string> {
  return { "X-Session-ID": getSessionId() };
}

export async function fetchJson<T>(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    let errorMessage = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json() as { error?: string; message?: string };
      errorMessage = payload.error || payload.message || errorMessage;
    } catch {
      // Fall back to the HTTP status when the response body is not JSON.
    }
    throw new Error(errorMessage);
  }
  return response.json() as Promise<T>;
}

// =====================================================
// KONFIGURACE MODELŮ
// =====================================================
export const AVAILABLE_MODELS = [
  { id: "qwen2.5vl:7b", name: "Qwen 2.5 VL", type: "local" as const },
  { id: "llava:7b", name: "Llava v1.6 7B", type: "local" as const },
  { id: "llama3.2-vision", name: "Llama 3.2 Vision", type: "local" as const },
];

// =====================================================
// TYPY
// =====================================================
export type FileType = "text" | "image" | "video" | "archive";
export type FeatureSpecValue = [number, number] | string[];
export type FeatureSpec = Record<string, FeatureSpecValue>;

export interface ProcessingResult {
  type?: FileType;
  status?: string;
  description?: string;
  job_id?: string;
  outputs?: Record<string, unknown>;
  key_frame_analysis?: {
    outputs?: Record<string, unknown>;
    transcription?: string;
  };
  feature_specification?: string | Record<string, unknown>;
  tabular_output?: Record<
    string,
    { transcript?: string; [k: string]: unknown }
  >;
  transcriptions?: Record<string, string>;
  transcription?: string;
  transcript?: string;
  asr?: { text?: string };
  [key: string]: unknown;
}

export interface BackendOk {
  message: string;
  files: string[];
  processing: ProcessingResult;
}

// =====================================================
// ML PIPELINE TYPY
// =====================================================

export interface ExtractDetails {
  status: string;
  dataset_type: string;
  dataset_X: Record<string, unknown>[];
  feature_spec: FeatureSpec;
  rows_count: number;
  dataset_Y_columns?: string[];
  clamped_count?: number;
}

export interface TrainResult {
  status: string;
  target_mode?: TargetMode;
  mse?: number;
  rulekit_mse?: number;
  xgb_mse?: number;
  cv_mse?: number;
  cv_std?: number;
  cv_mae?: number;
  train_accuracy?: number;
  train_balanced_accuracy?: number;
  train_f1_macro?: number;
  train_mcc?: number;
  cv_accuracy?: number;
  cv_balanced_accuracy?: number;
  cv_f1_macro?: number;
  cv_precision_macro?: number;
  cv_recall_macro?: number;
  cv_mcc?: number;
  cv_folds?: number;
  rules_count?: number;
  rules?: string[];
  feature_spec?: FeatureSpec;
  feature_importance?: {
    xgboost?: Record<string, number>;
    rulekit?: Record<string, number>;
  };
  warnings?: string[];
  training_data_X?: Record<string, unknown>[];
  error?: string;
}

export interface PipelineState {
  feature_spec: FeatureSpec;
  target_variable: string;
  target_mode?: TargetMode;
  is_trained: boolean;
  completed_phases: number[];
  suggested_step: number;
  training_rows: number;
  testing_rows: number;
  training_data_X: Record<string, unknown>[] | null;
  testing_data_X: Record<string, unknown>[] | null;
  dataset_Y_columns: string[] | null;
  train_result: TrainResult | null;
}

export interface PredictionItem {
  media_name: string;
  predicted_score?: number;
  actual_score?: number;
  predicted_label?: string;
  actual_label?: string;
  confidence?: number | null;
  rule_applied: string;
  extracted_features: Record<string, unknown>;
}

export interface PredictionMetrics {
  mode?: TargetMode;
  mse?: number;
  mae?: number;
  correlation?: number | null;
  accuracy?: number;
  balanced_accuracy?: number;
  f1_macro?: number;
  precision_macro?: number;
  recall_macro?: number;
  mcc?: number;
  labels?: string[];
  confusion_matrix?: number[][];
  class_metrics?: Array<{
    label: string;
    precision: number;
    recall: number;
    f1: number;
    support: number;
  }>;
  avg_confidence?: number | null;
  correct_confidence_avg?: number | null;
  incorrect_confidence_avg?: number | null;
  matched_count: number;
  total_count: number;
}

export type TargetMode = "regression" | "classification";

export interface PredictDetails {
  status: string;
  predictions: PredictionItem[];
  metrics: PredictionMetrics | null;
  count: number;
}

export type StatusPayload = {
  progress: number;
  stage?: string;
  done?: boolean;
  error?: string;
  details?: ExtractDetails | TrainResult | PredictDetails;
  suggested_features?: FeatureSpec;
};

// =====================================================
// STYLE KONSTANTY
// =====================================================
export const TYPE_STYLES: Record<
  FileType,
  { chip: string; accent: string; border: string; label: string }
> = {
  image: {
    chip: "bg-blue-50 text-blue-900",
    accent: "bg-blue-500",
    border: "border-blue-300",
    label: "Obrázky",
  },
  video: {
    chip: "bg-green-50 text-green-900",
    accent: "bg-green-500",
    border: "border-green-300",
    label: "Video",
  },
  text: {
    chip: "bg-red-50 text-red-900",
    accent: "bg-red-500",
    border: "border-red-300",
    label: "PDF",
  },
  archive: {
    chip: "bg-amber-50 text-amber-900",
    accent: "bg-amber-500",
    border: "border-amber-300",
    label: "ZIP",
  },
};

export const EXT_GROUPS: Record<FileType, string[]> = {
  text: [".pdf", ".txt", ".md", ".csv"],
  image: [".png", ".jpg", ".jpeg", ".webp", ".heic", ".gif"],
  video: [".mp4", ".avi", ".mov", ".mkv"],
  archive: [".zip"],
};
