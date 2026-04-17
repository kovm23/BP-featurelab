import type {
  FeatureSpec,
  PredictionItem,
  PredictionMetrics,
  StatusPayload,
  TargetMode,
  TrainResult,
} from "@/lib/api";
import {
  DISCOVER_URL,
  EXTRACT_LOCAL_URL,
  EXTRACT_URL,
  PREDICT_URL,
  TRAIN_URL,
  sessionHeaders,
} from "@/lib/api";
import { pollProgress, type PollReason } from "@/hooks/usePollProgress";

export interface ExtractRequestConfig {
  datasetType: "training" | "testing";
  modelProvider: string;
  featureSpec: FeatureSpec | null;
  zipFile?: File;
  labelsFile?: File | null;
  zipPath?: string;
  labelsPath?: string;
}

async function parseError(response: Response, fallbackMessage: string) {
  const errData = await response.json().catch(() => ({}));
  return (errData as { error?: string }).error || fallbackMessage;
}

export async function submitDiscoveryRequest(params: {
  sampleFiles: File[];
  labelsFile?: File | null;
  targetVariable: string;
  targetMode: TargetMode;
  modelProvider: string;
}) {
  const formData = new FormData();
  for (const file of params.sampleFiles) formData.append("files", file, file.name);
  formData.append("target_variable", params.targetVariable);
  formData.append("target_mode", params.targetMode);
  formData.append("model", params.modelProvider);
  if (params.labelsFile) formData.append("labels_file", params.labelsFile);

  const response = await fetch(DISCOVER_URL, {
    method: "POST",
    body: formData,
    headers: sessionHeaders(),
  });
  if (!response.ok) {
    throw new Error(await parseError(response, "Feature Discovery failed"));
  }
  return response.json();
}

export async function pollDiscoveryJob(params: {
  jobId: string;
  signal: AbortSignal;
  onProgress: (payload: StatusPayload) => void;
  onReconnecting?: (attempts: number) => void;
}): Promise<PollReason> {
  return pollProgress(params.jobId, params.onProgress, params.signal, params.onReconnecting);
}

export async function submitExtractRequest(config: ExtractRequestConfig) {
  if (config.zipFile) {
    const formData = new FormData();
    formData.append("file", config.zipFile);
    formData.append("model", config.modelProvider);
    formData.append("feature_spec", JSON.stringify(config.featureSpec));
    formData.append("dataset_type", config.datasetType);
    if (config.labelsFile) formData.append("labels_file", config.labelsFile);

    const response = await fetch(EXTRACT_URL, {
      method: "POST",
      body: formData,
      headers: sessionHeaders(),
    });
    if (!response.ok) {
      throw new Error(await parseError(response, "Extraction failed"));
    }
    return response.json();
  }

  const response = await fetch(EXTRACT_LOCAL_URL, {
    method: "POST",
    headers: { ...sessionHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({
      zip_path: config.zipPath,
      labels_path: config.labelsPath || undefined,
      model: config.modelProvider,
      feature_spec: config.featureSpec,
      dataset_type: config.datasetType,
    }),
  });
  if (!response.ok) {
    throw new Error(await parseError(response, "Extraction failed"));
  }
  return response.json();
}

export async function pollExtractJob(params: {
  jobId: string;
  signal: AbortSignal;
  onProgress: (payload: StatusPayload) => void;
  onReconnecting?: (attempts: number) => void;
}): Promise<PollReason> {
  return pollProgress(params.jobId, params.onProgress, params.signal, params.onReconnecting);
}

export async function submitTrainRequest(params: {
  targetColumn: string;
  targetMode: TargetMode;
  signal: AbortSignal;
}) {
  const response = await fetch(TRAIN_URL, {
    method: "POST",
    headers: { ...sessionHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ target_column: params.targetColumn, target_mode: params.targetMode }),
    signal: params.signal,
  });
  if (!response.ok) {
    throw new Error(await parseError(response, "Training failed"));
  }
  return response.json() as Promise<{ job_id: string }>;
}

export async function pollTrainJob(params: {
  jobId: string;
  signal: AbortSignal;
  onProgress: (payload: StatusPayload) => void;
  onReconnecting?: (attempts: number) => void;
}): Promise<PollReason> {
  return pollProgress(params.jobId, params.onProgress, params.signal, params.onReconnecting);
}

export async function submitPredictRequest(params: {
  labelsFile?: File | null;
  signal: AbortSignal;
}) {
  let response: Response;
  if (params.labelsFile) {
    const formData = new FormData();
    formData.append("labels_file", params.labelsFile);
    response = await fetch(PREDICT_URL, {
      method: "POST",
      body: formData,
      headers: sessionHeaders(),
      signal: params.signal,
    });
  } else {
    response = await fetch(PREDICT_URL, {
      method: "POST",
      headers: sessionHeaders(),
      signal: params.signal,
    });
  }
  if (!response.ok) {
    throw new Error(await parseError(response, "Prediction failed"));
  }
  return response.json() as Promise<{ job_id: string }>;
}

export async function pollPredictJob(params: {
  jobId: string;
  signal: AbortSignal;
  onProgress: (payload: StatusPayload) => void;
  onReconnecting?: (attempts: number) => void;
}): Promise<PollReason> {
  return pollProgress(params.jobId, params.onProgress, params.signal, params.onReconnecting);
}

export type TrainPollResult = StatusPayload & { details?: TrainResult };
export type PredictPollResult = StatusPayload & {
  details?: { predictions: PredictionItem[]; metrics: PredictionMetrics };
};
