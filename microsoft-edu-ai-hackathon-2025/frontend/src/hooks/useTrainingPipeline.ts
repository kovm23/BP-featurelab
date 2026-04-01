/**
 * Custom hook that encapsulates all training pipeline state and handlers.
 * Extracted from App.tsx to reduce component complexity.
 */
import { useState, useMemo, useEffect, useRef } from "react";
import type {
  FeatureSpec,
  PipelineState,
  StatusPayload,
  TargetMode,
  TrainResult,
  PredictionItem,
  PredictionMetrics,
} from "@/lib/api";
import {
  DISCOVER_URL,
  EXTRACT_URL,
  EXTRACT_LOCAL_URL,
  fetchJson,
  TRAIN_URL,
  PREDICT_URL,
  RESET_URL,
  STATE_URL,
  STATUS_URL,
  HEALTH_URL,
  QUEUE_INFO_URL,
  sessionHeaders,
} from "@/lib/api";
import { pollProgress } from "@/hooks/usePollProgress";

/* ------------------------------------------------------------------ */
/*  Persisted state shape (localStorage)                               */
/* ------------------------------------------------------------------ */
interface PersistedPipeline {
  trainingStep?: 1 | 2 | 3 | 4 | 5;
  targetVariable?: string;
  targetMode?: TargetMode;
  featureSpec?: FeatureSpec | null;
  modelProvider?: string;
}

const STORAGE_KEY = "mflPipeline";

function loadPersisted(): PersistedPipeline | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function savePersisted(data: PersistedPipeline) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {
    /* quota exceeded or localStorage unavailable */
  }
}

/* ------------------------------------------------------------------ */
/*  Active job persistence (resume polling after page reload)          */
/* ------------------------------------------------------------------ */
type ActiveJobPhase = "discover" | "extract_training" | "extract_testing";

interface ActiveJob {
  job_id: string;
  phase: ActiveJobPhase;
}

const ACTIVE_JOB_KEY = "mflActiveJob";

function saveActiveJob(job: ActiveJob) {
  try {
    localStorage.setItem(ACTIVE_JOB_KEY, JSON.stringify(job));
  } catch { /* */ }
}

function clearActiveJob() {
  try {
    localStorage.removeItem(ACTIVE_JOB_KEY);
  } catch { /* */ }
}

function loadActiveJob(): ActiveJob | null {
  try {
    const raw = localStorage.getItem(ACTIVE_JOB_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/*  Generic extraction helper                                          */
/* ------------------------------------------------------------------ */
interface ExtractConfig {
  url: string;
  body: BodyInit | string;
  headers?: Record<string, string>;
  datasetType: "training" | "testing";
  phaseLabel: string;
}

/* ------------------------------------------------------------------ */
/*  Hook                                                               */
/* ------------------------------------------------------------------ */
export function useTrainingPipeline(uiLanguage: "cs" | "en" = "cs") {
  const tx = uiLanguage === "en"
    ? {
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
    }
    : {
      restoring: "Obnovuji...",
      phase1Failed: "Fáze 1 selhala",
      phaseFailed: "selhala",
      extractFailed: "Extrakce selhala",
      featureDiscoveryFailed: "Feature Discovery selhala",
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
    };
  const saved = useMemo(loadPersisted, []);

  // --- Phase navigation ---
  const [trainingStep, setTrainingStep] = useState<1 | 2 | 3 | 4 | 5>(
    saved?.trainingStep ?? 1,
  );

  // --- Phase 1: Discovery ---
  const [targetVariable, setTargetVariable] = useState(
    saved?.targetVariable ?? "movie memorability score",
  );
  const [targetMode, setTargetMode] = useState<TargetMode>(
    saved?.targetMode ?? "regression",
  );
  const [featureSpec, setFeatureSpec] = useState<FeatureSpec | null>(
    saved?.featureSpec ?? null,
  );
  const [isDiscovering, setIsDiscovering] = useState(false);

  // --- Phase 2: Training extraction ---
  const [extractionBusy, setExtractionBusy] = useState(false);
  const [trainingDataX, setTrainingDataX] = useState<Record<string, unknown>[] | null>(null);
  const [datasetYColumns, setDatasetYColumns] = useState<string[] | null>(null);

  // --- Phase 3: Training ---
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);

  // --- Phase 4: Test extraction ---
  const [testExtractionBusy, setTestExtractionBusy] = useState(false);
  const [testingDataX, setTestingDataX] = useState<Record<string, unknown>[] | null>(null);

  // --- Phase 5: Prediction ---
  const [predictBusy, setPredictBusy] = useState(false);
  const [predictions, setPredictions] = useState<PredictionItem[] | null>(null);
  const [predictionMetrics, setPredictionMetrics] = useState<PredictionMetrics | null>(null);

  // --- Shared ---
  const [activeCtrl, setActiveCtrl] = useState<AbortController | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [modelProvider, setModelProvider] = useState<string>(
    saved?.modelProvider ?? "qwen2.5vl:7b",
  );

  // --- Ollama availability ---
  const [ollamaOk, setOllamaOk] = useState<boolean | null>(null);

  // --- Queue info (other users waiting) ---
  const [queueBusy, setQueueBusy] = useState(false);
  const [queuedCount, setQueuedCount] = useState(0);

  // Refs for values captured in async polling callbacks (prevents stale closures)
  const modelProviderRef = useRef(modelProvider);
  const targetVariableRef = useRef(targetVariable);
  const targetModeRef = useRef(targetMode);
  useEffect(() => { modelProviderRef.current = modelProvider; }, [modelProvider]);
  useEffect(() => { targetVariableRef.current = targetVariable; }, [targetVariable]);
  useEffect(() => { targetModeRef.current = targetMode; }, [targetMode]);

  // --- Restore from backend on mount ---
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;

    let cancelled = false;
    fetchJson<PipelineState>(STATE_URL, { headers: sessionHeaders() })
      .then((state: PipelineState) => {
        if (cancelled) return;
        // Only restore if backend has meaningful state
        if (!state.completed_phases || state.completed_phases.length === 0) return;

        if (state.feature_spec && Object.keys(state.feature_spec).length > 0) {
          setFeatureSpec(state.feature_spec);
        }
        if (state.target_variable) {
          setTargetVariable(state.target_variable);
        }
        if (state.target_mode) {
          setTargetMode(state.target_mode);
        }
        if (state.training_data_X && state.training_data_X.length > 0) {
          setTrainingDataX(state.training_data_X);
        }
        if (state.dataset_Y_columns) {
          setDatasetYColumns(state.dataset_Y_columns);
        }
        if (state.train_result) {
          setTrainResult(state.train_result);
        }
        if (state.testing_data_X && state.testing_data_X.length > 0) {
          setTestingDataX(state.testing_data_X);
        }
        // Advance step only if localStorage didn't already set a later step
        // Backend suggested_step is the source of truth when > current step
        setTrainingStep((prev) => {
          const suggested = Math.min(
            state.suggested_step as 1 | 2 | 3 | 4 | 5,
            5,
          ) as 1 | 2 | 3 | 4 | 5;
          return suggested > prev ? suggested : prev;
        });
      })
      .catch(() => {
        /* backend unreachable — continue with localStorage state */
      });
    return () => { cancelled = true; };
  }, []);

  // --- Resume active job after page reload ---
  const jobRestoredRef = useRef(false);
  useEffect(() => {
    if (jobRestoredRef.current) return;
    jobRestoredRef.current = true;

    const saved = loadActiveJob();
    if (!saved) return;

    // Check if job is still running
    fetchJson<StatusPayload>(STATUS_URL(saved.job_id), {
      cache: "no-store",
      headers: sessionHeaders(),
    })
      .then((s: StatusPayload) => {
        if (s.done) {
          // Job finished while we were away — clean up; state restore handled by /state effect
          clearActiveJob();
          return;
        }
        // Job still running — restore busy state and resume polling
        const ctrl = new AbortController();
        setActiveCtrl(ctrl);
        setProgress(Math.max(0, Math.min(100, s.progress ?? 0)));
        setProgressLabel(s.stage || tx.restoring);

        if (saved.phase === "discover") {
          setIsDiscovering(true);
          pollProgress(
            saved.job_id,
            (tick) => {
              setProgress(Math.max(0, Math.min(100, tick.progress ?? 0)));
              setProgressLabel(tick.stage || "");
              if (tick.done && !tick.error) {
                clearActiveJob();
                if (tick.suggested_features) {
                  setFeatureSpec(tick.suggested_features);
                  savePersisted({
                    trainingStep: 2,
                    targetVariable: targetVariableRef.current,
                    targetMode: targetModeRef.current,
                    featureSpec: tick.suggested_features,
                    modelProvider: modelProviderRef.current,
                  });
                }
              }
              if (tick.done && tick.error) {
                clearActiveJob();
                setError(tx.phase1Failed + ": " + tick.error);
              }
            },
            ctrl.signal,
          ).finally(() => { setIsDiscovering(false); setActiveCtrl(null); });
        } else {
          const isTraining = saved.phase === "extract_training";
          const setBusy = isTraining ? setExtractionBusy : setTestExtractionBusy;
          const setData = isTraining ? setTrainingDataX : setTestingDataX;
          setBusy(true);
          pollProgress(
            saved.job_id,
            (tick: StatusPayload) => {
              setProgress(Math.max(0, Math.min(100, tick.progress ?? 0)));
              setProgressLabel(tick.stage || "");
              if (tick.done && tick.details?.status === "success") {
                clearActiveJob();
                setData(tick.details.dataset_X);
                if (isTraining) setDatasetYColumns(tick.details.dataset_Y_columns || null);
              }
              if (tick.done && tick.error) {
                clearActiveJob();
                setError(`${tx.extractFailed}: ${tick.error}`);
              }
            },
            ctrl.signal,
          ).finally(() => { setBusy(false); setActiveCtrl(null); });
        }
      })
      .catch(() => {
        clearActiveJob(); // status fetch failed — assume gone
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // --- Ollama health check on mount (with one retry after 5s) ---
  useEffect(() => {
    let cancelled = false;
    const checkHealth = (retryOnFail: boolean) => {
      fetchJson<{ ok: boolean; ollama: boolean }>(HEALTH_URL, { cache: "no-store" })
        .then((data: { ok: boolean; ollama: boolean }) => {
          if (!cancelled) setOllamaOk(data.ollama);
        })
        .catch(() => {
          // Network error = backend not yet up; keep null (unknown) and retry once
          if (!cancelled && retryOnFail) {
            setTimeout(() => checkHealth(false), 5000);
          }
        });
    };
    checkHealth(true);
    return () => { cancelled = true; };
  }, []);

  // --- Poll queue info while any phase is busy ---
  const anyBusy = isDiscovering || extractionBusy || trainingBusy || testExtractionBusy || predictBusy;
  useEffect(() => {
    if (!anyBusy) { setQueueBusy(false); setQueuedCount(0); return; }
    let cancelled = false;
    const poll = () => {
      fetchJson<{ busy: boolean; queued: number }>(QUEUE_INFO_URL, {
        cache: "no-store",
        headers: sessionHeaders(),
      })
        .then((d: { busy: boolean; queued: number }) => {
          if (!cancelled) { setQueueBusy(d.busy); setQueuedCount(d.queued); }
        })
        .catch(() => {
          if (!cancelled) {
            setQueueBusy(false);
            setQueuedCount(0);
          }
        });
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, [anyBusy]);

  // --- Persistence (lightweight — only metadata, no large datasets) ---
  function persist() {
    savePersisted({
      trainingStep,
      targetVariable,
      targetMode,
      featureSpec,
      modelProvider,
    });
  }

  /* ---------------------------------------------------------------- */
  /*  Cancel                                                           */
  /* ---------------------------------------------------------------- */
  function handleCancelActive() {
    if (activeCtrl) {
      activeCtrl.abort();
      setActiveCtrl(null);
    }
    clearActiveJob();
    setExtractionBusy(false);
    setTrainingBusy(false);
    setTestExtractionBusy(false);
    setPredictBusy(false);
    setIsDiscovering(false);
    setProgress(0);
    setProgressLabel("");
  }

  /* ---------------------------------------------------------------- */
  /*  Phase 1: Discovery                                               */
  /* ---------------------------------------------------------------- */
  async function handleDiscover(sampleFiles: File[], labelsFile?: File | null) {
    setIsDiscovering(true);
    setFeatureSpec(null);
    setError(null);

    const formData = new FormData();
    for (const f of sampleFiles) formData.append("files", f, f.name);
    formData.append("target_variable", targetVariable);
    formData.append("target_mode", targetMode);
    formData.append("model", modelProvider);
    if (labelsFile) formData.append("labels_file", labelsFile);

    try {
      const res = await fetch(DISCOVER_URL, { method: "POST", body: formData, headers: sessionHeaders() });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || tx.featureDiscoveryFailed);
      }
      const data = await res.json();
      if (data.job_id) {
        saveActiveJob({ job_id: data.job_id, phase: "discover" });
        const ctrl = new AbortController();
        setActiveCtrl(ctrl);
        await pollProgress(
          data.job_id,
          (s) => {
            setProgress(Math.max(0, Math.min(100, s.progress ?? 0)));
            setProgressLabel(s.stage || "");
            if (s.done && !s.error) {
              clearActiveJob();
              if (s.suggested_features) {
                setFeatureSpec(s.suggested_features);
                savePersisted({
                  trainingStep: 2,
                  targetVariable: targetVariableRef.current,
                  targetMode: targetModeRef.current,
                  featureSpec: s.suggested_features,
                  modelProvider: modelProviderRef.current,
                });
              }
            }
            if (s.done && s.error) {
              clearActiveJob();
              setError(tx.phase1Failed + ": " + s.error);
            }
          },
          ctrl.signal,
        );
      } else if (data.suggested_features) {
        setFeatureSpec(data.suggested_features);
      }
    } catch (err: unknown) {
      setError(tx.phase1Failed + ": " + (err instanceof Error ? err.message : String(err)));
    } finally {
      setIsDiscovering(false);
      setActiveCtrl(null);
    }
  }

  /* ---------------------------------------------------------------- */
  /*  Generic extraction (Phase 2 + 4)                                 */
  /* ---------------------------------------------------------------- */
  async function _runExtract(config: ExtractConfig) {
    const isTraining = config.datasetType === "training";
    const setBusy = isTraining ? setExtractionBusy : setTestExtractionBusy;
    const setData = isTraining ? setTrainingDataX : setTestingDataX;

    setBusy(true);
    setData(null);
    setProgress(0);
    setProgressLabel(tx.startingExtraction);
    setError(null);
    // Clear stale downstream results so user doesn't see old data after re-extraction
    if (isTraining) setTrainResult(null);
    else { setPredictions(null); setPredictionMetrics(null); }

    try {
      const res = await fetch(config.url, {
        method: "POST",
        body: config.body,
        headers: { ...sessionHeaders(), ...(config.headers || {}) },
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || tx.extractFailed);
      }
      const data = await res.json();

      if (data.job_id) {
        saveActiveJob({
          job_id: data.job_id,
          phase: isTraining ? "extract_training" : "extract_testing",
        });
        const ctrl = new AbortController();
        setActiveCtrl(ctrl);
        await pollProgress(
          data.job_id,
          (s: StatusPayload) => {
            setProgress(Math.max(0, Math.min(100, s.progress ?? 0)));
            setProgressLabel(s.stage || "");
            if (s.done && s.details?.status === "success") {
              clearActiveJob();
              setData(s.details.dataset_X);
              if (isTraining) setDatasetYColumns(s.details.dataset_Y_columns || null);
            }
            if (s.done && s.error) {
              clearActiveJob();
              setError(`${config.phaseLabel} ${tx.phaseFailed}: ${s.error}`);
            }
          },
          ctrl.signal,
        );
      }
    } catch (e: unknown) {
      setError(`${config.phaseLabel} ${tx.phaseFailed}: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  /* Phase 2: upload */
  async function handleExtractTraining(zipFile: File, labelsFile?: File | null) {
    const formData = new FormData();
    formData.append("file", zipFile);
    formData.append("model", modelProvider);
    formData.append("feature_spec", JSON.stringify(featureSpec));
    formData.append("dataset_type", "training");
    if (labelsFile) formData.append("labels_file", labelsFile);

    return _runExtract({
      url: EXTRACT_URL,
      body: formData,
      datasetType: "training",
      phaseLabel: tx.phase2,
    });
  }

  /* Phase 2: server path */
  async function handleExtractTrainingLocal(zipPath: string, labelsPath?: string) {
    return _runExtract({
      url: EXTRACT_LOCAL_URL,
      body: JSON.stringify({
        zip_path: zipPath,
        labels_path: labelsPath || undefined,
        model: modelProvider,
        feature_spec: featureSpec,
        dataset_type: "training",
      }),
      headers: { "Content-Type": "application/json" },
      datasetType: "training",
      phaseLabel: tx.phase2,
    });
  }

  /* Phase 4: upload */
  async function handleExtractTesting(zipFile: File) {
    const formData = new FormData();
    formData.append("file", zipFile);
    formData.append("model", modelProvider);
    formData.append("feature_spec", JSON.stringify(featureSpec));
    formData.append("dataset_type", "testing");

    return _runExtract({
      url: EXTRACT_URL,
      body: formData,
      datasetType: "testing",
      phaseLabel: tx.phase4,
    });
  }

  /* Phase 4: server path */
  async function handleExtractTestingLocal(zipPath: string) {
    return _runExtract({
      url: EXTRACT_LOCAL_URL,
      body: JSON.stringify({
        zip_path: zipPath,
        model: modelProvider,
        feature_spec: featureSpec,
        dataset_type: "testing",
      }),
      headers: { "Content-Type": "application/json" },
      datasetType: "testing",
      phaseLabel: tx.phase4,
    });
  }

  /* ---------------------------------------------------------------- */
  /*  Phase 3: Training                                                */
  /* ---------------------------------------------------------------- */
  async function handleTrain(targetColumn: string) {
    setTrainingBusy(true);
    setTrainResult(null);
    setError(null);
    setProgress(0);
    setProgressLabel(tx.startingTraining);

    const ctrl = new AbortController();
    setActiveCtrl(ctrl);

    try {
      const res = await fetch(TRAIN_URL, {
        method: "POST",
        headers: { ...sessionHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ target_column: targetColumn, target_mode: targetMode }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error((errData as { error?: string }).error || tx.trainingFailed);
      }
      const { job_id } = await res.json() as { job_id: string };
      await pollProgress(
        job_id,
        (tick) => {
          setProgress(Math.max(0, Math.min(100, tick.progress ?? 0)));
          setProgressLabel(tick.stage || tx.trainingInProgress);
          if (tick.done && !tick.error) {
            setTrainResult(tick.details as TrainResult);
          }
          if (tick.done && tick.error) {
            setError(tx.phase3Failed + ": " + tick.error);
          }
        },
        ctrl.signal,
      );
    } catch (e: unknown) {
      if (!ctrl.signal.aborted) {
        setError(tx.phase3Failed + ": " + (e instanceof Error ? e.message : String(e)));
      }
    } finally {
      setTrainingBusy(false);
      setActiveCtrl(null);
    }
  }

  /* ---------------------------------------------------------------- */
  /*  Phase 5: Prediction                                              */
  /* ---------------------------------------------------------------- */
  async function handlePredict(labelsFile?: File | null) {
    setPredictBusy(true);
    setPredictions(null);
    setPredictionMetrics(null);
    setError(null);
    setProgress(0);
    setProgressLabel(tx.startingPrediction);

    const ctrl = new AbortController();
    setActiveCtrl(ctrl);

    try {
      let res: Response;
      if (labelsFile) {
        const formData = new FormData();
        formData.append("labels_file", labelsFile);
        res = await fetch(PREDICT_URL, { method: "POST", body: formData, headers: sessionHeaders(), signal: ctrl.signal });
      } else {
        res = await fetch(PREDICT_URL, { method: "POST", headers: sessionHeaders(), signal: ctrl.signal });
      }
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error((errData as { error?: string }).error || tx.predictionFailed);
      }
      const { job_id } = await res.json() as { job_id: string };
      await pollProgress(
        job_id,
        (tick) => {
          setProgress(Math.max(0, Math.min(100, tick.progress ?? 0)));
          setProgressLabel(tick.stage || tx.predictingInProgress);
          if (tick.done && !tick.error) {
            const d = tick.details as { predictions: PredictionItem[]; metrics: PredictionMetrics };
            setPredictions(d.predictions);
            setPredictionMetrics(d.metrics || null);
          }
          if (tick.done && tick.error) {
            setError(tx.phase5Failed + ": " + tick.error);
          }
        },
        ctrl.signal,
      );
    } catch (e: unknown) {
      if (!ctrl.signal.aborted) {
        setError(tx.phase5Failed + ": " + (e instanceof Error ? e.message : String(e)));
      }
    } finally {
      setPredictBusy(false);
      setActiveCtrl(null);
    }
  }

  /* ---------------------------------------------------------------- */
  /*  Reset                                                            */
  /* ---------------------------------------------------------------- */
  function resetPipeline() {
    setTrainingStep(1);
    setTargetVariable("movie memorability score");
    setTargetMode("regression");
    setFeatureSpec(null);
    setIsDiscovering(false);
    setExtractionBusy(false);
    setTrainingDataX(null);
    setDatasetYColumns(null);
    setTrainingBusy(false);
    setTrainResult(null);
    setTestExtractionBusy(false);
    setTestingDataX(null);
    setPredictBusy(false);
    setPredictions(null);
    setPredictionMetrics(null);
    setProgress(0);
    setProgressLabel("");
    setError(null);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* */
    }
    clearActiveJob();
    fetch(RESET_URL, { method: "POST", headers: sessionHeaders() }).catch(() => {});
  }

  return {
    // Navigation
    trainingStep, setTrainingStep,
    // Phase 1
    targetVariable, setTargetVariable,
    targetMode, setTargetMode,
    featureSpec, setFeatureSpec,
    isDiscovering,
    handleDiscover,
    // Phase 2
    extractionBusy,
    trainingDataX,
    datasetYColumns,
    handleExtractTraining,
    handleExtractTrainingLocal,
    // Phase 3
    trainingBusy,
    trainResult,
    handleTrain,
    // Phase 4
    testExtractionBusy,
    testingDataX,
    handleExtractTesting,
    handleExtractTestingLocal,
    // Phase 5
    predictBusy,
    predictions,
    predictionMetrics,
    handlePredict,
    // Common
    modelProvider, setModelProvider,
    progress,
    progressLabel,
    error,
    clearError: () => setError(null),
    handleCancelActive,
    resetPipeline,
    persist,
    ollamaOk,
    queueBusy,
    queuedCount,
    recheckOllama: () => {
      fetchJson<{ ok: boolean; ollama: boolean }>(HEALTH_URL, { cache: "no-store" })
        .then((data: { ok: boolean; ollama: boolean }) => setOllamaOk(data.ollama))
        .catch(() => { /* keep current state */ });
    },
  };
}
