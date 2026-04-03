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
  fetchJson,
  RESET_URL,
  STATE_URL,
  STATUS_URL,
  sessionHeaders,
} from "@/lib/api";
import {
  clearActiveJob,
  clearPersisted,
  getTrainingPipelineText,
  loadActiveJob,
  loadPersisted,
  saveActiveJob,
  savePersisted,
} from "@/hooks/trainingPipelineUtils";
import {
  pollDiscoveryJob,
  pollExtractJob,
  pollPredictJob,
  pollTrainJob,
  submitDiscoveryRequest,
  submitExtractRequest,
  submitPredictRequest,
  submitTrainRequest,
} from "@/hooks/trainingPipelineRequests";
import {
  applyBackendPipelineState,
  persistDiscoveryOutcome,
} from "@/hooks/trainingPipelineRecovery";
import {
  clearActivePipelineRuntime,
  invalidatePipelineFromPhase,
  resetPipelineLocalState,
} from "@/hooks/trainingPipelineState";
import { usePipelineRuntime } from "@/hooks/usePipelineRuntime";

/* ------------------------------------------------------------------ */
/*  Hook                                                               */
/* ------------------------------------------------------------------ */
export function useTrainingPipeline(uiLanguage: "cs" | "en" = "cs") {
  const tx = getTrainingPipelineText(uiLanguage);
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

  const pipelineStateSetters = {
    setFeatureSpec,
    setTrainingDataX,
    setDatasetYColumns,
    setTrainResult,
    setTestingDataX,
    setPredictions,
    setPredictionMetrics,
    setTrainingStep,
  };

  const pipelineRuntimeSetters = {
    setTrainingStep,
    setTargetVariable,
    setTargetMode,
    setFeatureSpec,
    setIsDiscovering,
    setExtractionBusy,
    setTrainingDataX,
    setDatasetYColumns,
    setTrainingBusy,
    setTrainResult,
    setTestExtractionBusy,
    setTestingDataX,
    setPredictBusy,
    setPredictions,
    setPredictionMetrics,
    setProgress,
    setProgressLabel,
    setError,
  };

  // Refs for values captured in async polling callbacks (prevents stale closures)
  const modelProviderRef = useRef(modelProvider);
  const targetVariableRef = useRef(targetVariable);
  const targetModeRef = useRef(targetMode);
  useEffect(() => { modelProviderRef.current = modelProvider; }, [modelProvider]);
  useEffect(() => { targetVariableRef.current = targetVariable; }, [targetVariable]);
  useEffect(() => { targetModeRef.current = targetMode; }, [targetMode]);

  useEffect(() => {
    savePersisted({
      trainingStep,
      targetVariable,
      targetMode,
      featureSpec,
      modelProvider,
    });
  }, [trainingStep, targetVariable, targetMode, featureSpec, modelProvider]);

  // --- Restore from backend on mount ---
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;

    let cancelled = false;
    fetchJson<PipelineState>(STATE_URL, { headers: sessionHeaders() })
      .then((state: PipelineState) => {
        if (cancelled) return;
        applyBackendPipelineState(state, {
          ...pipelineRuntimeSetters,
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
          pollDiscoveryJob({
            jobId: saved.job_id,
            signal: ctrl.signal,
            onProgress: (tick) => {
              setProgress(Math.max(0, Math.min(100, tick.progress ?? 0)));
              setProgressLabel(tick.stage || "");
              if (tick.done && !tick.error) {
                clearActiveJob();
                if (tick.suggested_features) {
                  setFeatureSpec(tick.suggested_features);
                  persistDiscoveryOutcome({
                    featureSpec: tick.suggested_features,
                    targetVariableRef,
                    targetModeRef,
                    modelProviderRef,
                  });
                }
              }
              if (tick.done && tick.error) {
                clearActiveJob();
                setError(tx.phase1Failed + ": " + tick.error);
              }
            },
          }).finally(() => { setIsDiscovering(false); setActiveCtrl(null); });
        } else if (saved.phase === "extract_training" || saved.phase === "extract_testing") {
          const isTraining = saved.phase === "extract_training";
          const setBusy = isTraining ? setExtractionBusy : setTestExtractionBusy;
          const setData = isTraining ? setTrainingDataX : setTestingDataX;
          setBusy(true);
          pollExtractJob({
            jobId: saved.job_id,
            signal: ctrl.signal,
            onProgress: (tick: StatusPayload) => {
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
          }).finally(() => { setBusy(false); setActiveCtrl(null); });
        } else if (saved.phase === "train") {
          setTrainingBusy(true);
          pollTrainJob({
            jobId: saved.job_id,
            signal: ctrl.signal,
            onProgress: (tick) => {
              setProgress(Math.max(0, Math.min(100, tick.progress ?? 0)));
              setProgressLabel(tick.stage || tx.trainingInProgress);
              if (tick.done && !tick.error) {
                clearActiveJob();
                setTrainResult(tick.details as TrainResult);
              }
              if (tick.done && tick.error) {
                clearActiveJob();
                setError(tx.phase3Failed + ": " + tick.error);
              }
            },
          }).finally(() => { setTrainingBusy(false); setActiveCtrl(null); });
        } else if (saved.phase === "predict") {
          setPredictBusy(true);
          pollPredictJob({
            jobId: saved.job_id,
            signal: ctrl.signal,
            onProgress: (tick) => {
              setProgress(Math.max(0, Math.min(100, tick.progress ?? 0)));
              setProgressLabel(tick.stage || tx.predictingInProgress);
              if (tick.done && !tick.error) {
                clearActiveJob();
                const d = tick.details as { predictions: PredictionItem[]; metrics: PredictionMetrics };
                setPredictions(d.predictions);
                setPredictionMetrics(d.metrics || null);
              }
              if (tick.done && tick.error) {
                clearActiveJob();
                setError(tx.phase5Failed + ": " + tick.error);
              }
            },
          }).finally(() => { setPredictBusy(false); setActiveCtrl(null); });
        }
      })
      .catch(() => {
        clearActiveJob(); // status fetch failed — assume gone
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const anyBusy = isDiscovering || extractionBusy || trainingBusy || testExtractionBusy || predictBusy;
  const { ollamaOk, queueBusy, queuedCount, recheckOllama } = usePipelineRuntime(anyBusy);

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
    clearActivePipelineRuntime({
      setExtractionBusy,
      setTrainingBusy,
      setTestExtractionBusy,
      setPredictBusy,
      setIsDiscovering,
      setProgress,
      setProgressLabel,
    });
  }

  /* ---------------------------------------------------------------- */
  /*  Phase 1: Discovery                                               */
  /* ---------------------------------------------------------------- */
  function updateTargetVariable(nextValue: string) {
    if (nextValue !== targetVariable && (featureSpec || trainingDataX || trainResult || testingDataX || predictions)) {
      invalidatePipelineFromPhase(pipelineStateSetters, 1);
    }
    setTargetVariable(nextValue);
  }

  function updateTargetMode(nextMode: TargetMode) {
    if (nextMode !== targetMode && (featureSpec || trainingDataX || trainResult || testingDataX || predictions)) {
      invalidatePipelineFromPhase(pipelineStateSetters, 1);
    }
    setTargetMode(nextMode);
  }

  function updateFeatureSpec(nextSpec: FeatureSpec) {
    const changed = JSON.stringify(featureSpec ?? {}) !== JSON.stringify(nextSpec ?? {});
    if (changed && (trainingDataX || trainResult || testingDataX || predictions)) {
      invalidatePipelineFromPhase(pipelineStateSetters, 2);
    }
    setFeatureSpec(nextSpec);
  }

  async function handleDiscover(sampleFiles: File[], labelsFile?: File | null) {
    invalidatePipelineFromPhase(pipelineStateSetters, 1);
    setIsDiscovering(true);
    setError(null);
    setProgress(0);
    setProgressLabel("");

    try {
      const data = await submitDiscoveryRequest({
        sampleFiles,
        labelsFile,
        targetVariable,
        targetMode,
        modelProvider,
      });
      if (data.job_id) {
        saveActiveJob({ job_id: data.job_id, phase: "discover" });
        const ctrl = new AbortController();
        setActiveCtrl(ctrl);
        await pollDiscoveryJob({
          jobId: data.job_id,
          signal: ctrl.signal,
          onProgress: (s) => {
            setProgress(Math.max(0, Math.min(100, s.progress ?? 0)));
            setProgressLabel(s.stage || "");
            if (s.done && !s.error) {
              clearActiveJob();
              if (s.suggested_features) {
                setFeatureSpec(s.suggested_features);
                persistDiscoveryOutcome({
                  featureSpec: s.suggested_features,
                  targetVariableRef,
                  targetModeRef,
                  modelProviderRef,
                });
              }
            }
            if (s.done && s.error) {
              clearActiveJob();
              setError(tx.phase1Failed + ": " + s.error);
            }
          },
        });
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
  async function _runExtract(config: {
    datasetType: "training" | "testing";
    phaseLabel: string;
    zipFile?: File;
    labelsFile?: File | null;
    zipPath?: string;
    labelsPath?: string;
  }) {
    const isTraining = config.datasetType === "training";
    const setBusy = isTraining ? setExtractionBusy : setTestExtractionBusy;
    const setData = isTraining ? setTrainingDataX : setTestingDataX;

    invalidatePipelineFromPhase(pipelineStateSetters, isTraining ? 2 : 4);
    setBusy(true);
    setProgress(0);
    setProgressLabel(tx.startingExtraction);
    setError(null);

    try {
      const data = await submitExtractRequest({
        datasetType: config.datasetType,
        modelProvider,
        featureSpec,
        zipFile: config.zipFile,
        labelsFile: config.labelsFile,
        zipPath: config.zipPath,
        labelsPath: config.labelsPath,
      });

      if (data.job_id) {
        saveActiveJob({
          job_id: data.job_id,
          phase: isTraining ? "extract_training" : "extract_testing",
        });
        const ctrl = new AbortController();
        setActiveCtrl(ctrl);
        await pollExtractJob({
          jobId: data.job_id,
          signal: ctrl.signal,
          onProgress: (s: StatusPayload) => {
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
        });
      }
    } catch (e: unknown) {
      setError(`${config.phaseLabel} ${tx.phaseFailed}: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  /* Phase 2: upload */
  async function handleExtractTraining(zipFile: File, labelsFile?: File | null) {
    return _runExtract({
      datasetType: "training",
      phaseLabel: tx.phase2,
      zipFile,
      labelsFile,
    });
  }

  /* Phase 2: server path */
  async function handleExtractTrainingLocal(zipPath: string, labelsPath?: string) {
    return _runExtract({
      datasetType: "training",
      phaseLabel: tx.phase2,
      zipPath,
      labelsPath,
    });
  }

  /* Phase 4: upload */
  async function handleExtractTesting(zipFile: File) {
    return _runExtract({
      datasetType: "testing",
      phaseLabel: tx.phase4,
      zipFile,
    });
  }

  /* Phase 4: server path */
  async function handleExtractTestingLocal(zipPath: string) {
    return _runExtract({
      datasetType: "testing",
      phaseLabel: tx.phase4,
      zipPath,
    });
  }

  /* ---------------------------------------------------------------- */
  /*  Phase 3: Training                                                */
  /* ---------------------------------------------------------------- */
  async function handleTrain(targetColumn: string) {
    setTrainingBusy(true);
    setTrainResult(null);
    setPredictions(null);
    setPredictionMetrics(null);
    setError(null);
    setProgress(0);
    setProgressLabel(tx.startingTraining);

    const ctrl = new AbortController();
    setActiveCtrl(ctrl);

    try {
      const { job_id } = await submitTrainRequest({
        targetColumn,
        targetMode,
        signal: ctrl.signal,
      });
      saveActiveJob({ job_id, phase: "train" });
      await pollTrainJob({
        jobId: job_id,
        signal: ctrl.signal,
        onProgress: (tick) => {
          setProgress(Math.max(0, Math.min(100, tick.progress ?? 0)));
          setProgressLabel(tick.stage || tx.trainingInProgress);
          if (tick.done && !tick.error) {
            clearActiveJob();
            setTrainResult(tick.details as TrainResult);
          }
          if (tick.done && tick.error) {
            clearActiveJob();
            setError(tx.phase3Failed + ": " + tick.error);
          }
        },
      });
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
      const { job_id } = await submitPredictRequest({
        labelsFile,
        signal: ctrl.signal,
      });
      saveActiveJob({ job_id, phase: "predict" });
      await pollPredictJob({
        jobId: job_id,
        signal: ctrl.signal,
        onProgress: (tick) => {
          setProgress(Math.max(0, Math.min(100, tick.progress ?? 0)));
          setProgressLabel(tick.stage || tx.predictingInProgress);
          if (tick.done && !tick.error) {
            clearActiveJob();
            const d = tick.details as { predictions: PredictionItem[]; metrics: PredictionMetrics };
            setPredictions(d.predictions);
            setPredictionMetrics(d.metrics || null);
          }
          if (tick.done && tick.error) {
            clearActiveJob();
            setError(tx.phase5Failed + ": " + tick.error);
          }
        },
      });
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
    resetPipelineLocalState(pipelineRuntimeSetters);
    clearPersisted();
    clearActiveJob();
    fetch(RESET_URL, { method: "POST", headers: sessionHeaders() }).catch(() => {});
  }

  return {
    // Navigation
    trainingStep, setTrainingStep,
    // Phase 1
    targetVariable, setTargetVariable: updateTargetVariable,
    targetMode, setTargetMode: updateTargetMode,
    featureSpec, setFeatureSpec: updateFeatureSpec,
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
    recheckOllama,
  };
}
