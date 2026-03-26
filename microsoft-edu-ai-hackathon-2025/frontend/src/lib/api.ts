// =====================================================
// API URL KONFIGURACE
// =====================================================
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export const DISCOVER_URL = `${API_BASE}/discover`;
export const EXTRACT_URL = `${API_BASE}/extract`;
export const TRAIN_URL = `${API_BASE}/train`;
export const PREDICT_URL = `${API_BASE}/predict`;
export const ANALYZE_URL = `${API_BASE}/analyze`;
export const EXTRACT_LOCAL_URL = `${API_BASE}/extract-local`;
export const RESET_URL = `${API_BASE}/reset`;
export const STATE_URL = `${API_BASE}/state`;
export const HEALTH_URL = `${API_BASE}/health`;
export const STATUS_URL = (jobId: string) =>
  `${API_BASE}/status/${encodeURIComponent(jobId)}`;

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
  feature_spec: Record<string, string>;
  rows_count: number;
  dataset_Y_columns?: string[];
  clamped_count?: number;
}

export interface TrainResult {
  status: string;
  mse?: number;
  rulekit_mse?: number;
  xgb_mse?: number;
  cv_mse?: number;
  cv_std?: number;
  cv_mae?: number;
  cv_folds?: number;
  rules_count?: number;
  rules?: string[];
  feature_spec?: Record<string, string>;
  feature_importance?: {
    xgboost?: Record<string, number>;
    rulekit?: Record<string, number>;
  };
  warnings?: string[];
  training_data_X?: Record<string, unknown>[];
  error?: string;
}

export interface PipelineState {
  feature_spec: Record<string, string>;
  target_variable: string;
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
  predicted_score: number;
  actual_score?: number;
  rule_applied: string;
  extracted_features: Record<string, unknown>;
}

export interface PredictionMetrics {
  mse: number;
  mae: number;
  correlation: number | null;
  matched_count: number;
  total_count: number;
}

export type StatusPayload = {
  progress: number;
  stage?: string;
  done?: boolean;
  error?: string;
  details?: ExtractDetails;
  suggested_features?: Record<string, string>;
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
