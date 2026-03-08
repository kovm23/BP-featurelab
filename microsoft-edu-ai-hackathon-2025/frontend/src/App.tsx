import React, { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { useDropzone } from "react-dropzone";
import {
  Card,
  CardHeader,
  CardContent,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import {
  UploadCloud,
  X,
  Download,
  PlayCircle,
  Loader2,
  RefreshCw,
  HelpCircle,
  Moon,
  Sun,
  ChevronDown,
  Cpu,
} from "lucide-react";

// =====================================================
// IMPORTY z nových modulů
// =====================================================
import type {
  FileType,
  BackendOk,
  StatusPayload,
  ExtractDetails,
  TrainResult,
  PredictionItem,
  PredictionMetrics,
} from "@/lib/api";
import {
  AVAILABLE_MODELS,
  DISCOVER_URL,
  EXTRACT_URL,
  TRAIN_URL,
  PREDICT_URL,
  TYPE_STYLES,
} from "@/lib/api";
import {
  detectType,
  downloadText,
  downloadXLSX,
  humanSize,
  fileIcon,
  uploaderBorder,
  fileKey,
  mergeFiles,
  getOutputs,
  getProcType,
  getTranscript,
  tileBg,
  CopyButton,
  downloadTestingDataWithPrediction,
} from "@/lib/helpers";
import { pollProgress } from "@/hooks/usePollProgress";
import { Guide } from "@/components/Guide";
import { TrainingView } from "@/components/TrainingView";

// =====================================================
// HLAVNÍ KOMPONENTA
// =====================================================
export default function MediaFeatureLabPro() {
  const [appMode, setAppMode] = useState<"predict" | "train">("train");

  // --- Training pipeline state ---
  const [trainingStep, setTrainingStep] = useState<1 | 2 | 3 | 4 | 5>(1);
  const [targetVariable, setTargetVariable] = useState("movie memorability score");
  const [featureSpec, setFeatureSpec] = useState<Record<string, string> | null>(null);
  const [isDiscovering, setIsDiscovering] = useState(false);

  // Phase 2: extraction
  const [extractionBusy, setExtractionBusy] = useState(false);
  const [trainingDataX, setTrainingDataX] = useState<Record<string, unknown>[] | null>(null);
  const [datasetYColumns, setDatasetYColumns] = useState<string[] | null>(null);

  // Phase 3: training
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);

  // Phase 4: test extraction
  const [testExtractionBusy, setTestExtractionBusy] = useState(false);
  const [testingDataX, setTestingDataX] = useState<Record<string, unknown>[] | null>(null);

  // Phase 5: predictions
  const [predictBusy, setPredictBusy] = useState(false);
  const [predictions, setPredictions] = useState<PredictionItem[] | null>(null);
  const [predictionMetrics, setPredictionMetrics] = useState<PredictionMetrics | null>(null);

  const [files, setFiles] = useState<File[]>([]);
  const [fileType, setFileType] = useState<FileType | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [response, setResponse] = useState<BackendOk | null>(null);
  const [predictionResult, setPredictionResult] = useState<{
    score: number;
    rule_applied: string;
    extracted_features: Record<string, unknown>;
    testing_data_X: Record<string, unknown>;
  } | null>(null);

  const [description, setDescription] = useState("");
  const [categories, setCategories] = useState("");

  const [formats, setFormats] = useState<{
    json: boolean;
    csv: boolean;
    xlsx: boolean;
    xml: boolean;
  }>({
    json: true,
    csv: false,
    xlsx: false,
    xml: false,
  });

  const [deluxe, setDeluxe] = useState<boolean>(() => {
    const saved = localStorage.getItem("mflTheme");
    if (saved === "dark") return true;
    if (saved === "light") return false;
    return false;
  });

  const [modelProvider, setModelProvider] = useState<string>(
    AVAILABLE_MODELS[1].id
  );

  useEffect(() => {
    localStorage.setItem("mflTheme", deluxe ? "dark" : "light");
  }, [deluxe]);

  const [showGuide, setShowGuide] = useState(false);

  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState<string>("");
  const [hasRealProgress, setHasRealProgress] = useState(false);

  const [step, setStep] = useState<1 | 2 | 3>(1);

  const [uploadCtrl, setUploadCtrl] = useState<AbortController | null>(null);

  const onDrop = useCallback(
    (accepted: File[]) => {
      setError(null);
      setResponse(null);
      if (!accepted.length) return;

      const newType = detectType(accepted);
      if (!newType) {
        setError(
          "Nahraj pouze jeden typ souborů (PDF / Obrázky / Video / ZIP) a správné přípony."
        );
        return;
      }
      if (fileType && fileType !== newType) {
        setError(
          `Už máš rozpracovaný výběr typu "${fileType}". Přidej další soubory stejného typu nebo výběr resetuj.`
        );
        return;
      }

      setFiles((prev) => mergeFiles(prev, accepted));
      if (!fileType) setFileType(newType);

      setStep(1);
      setProgress(0);
      setProgressLabel("");
      setHasRealProgress(false);
    },
    [fileType]
  );

  useEffect(() => {
    try {
      const meta = files.map((f) => ({
        name: f.name,
        size: f.size,
        lastModified: f.lastModified,
        type: f.type,
      }));
      localStorage.setItem("mflFilesMeta", JSON.stringify(meta));
      if (fileType) localStorage.setItem("mflFileType", fileType);
      else localStorage.removeItem("mflFileType");
    } catch {
      /* localStorage unavailable */
    }
  }, [files, fileType]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: true,
    accept: {
      "application/pdf": [".pdf"],
      "image/*": [".png", ".jpg", ".jpeg", ".webp", ".heic", ".gif"],
      "video/*": [".mp4", ".avi", ".mov", ".mkv"],
      "application/zip": [".zip"],
      "text/*": [".txt", ".csv", ".md"],
    },
  });

  const outputFormatsString = useMemo(() => {
    const selected: string[] = [];
    if (formats.json) selected.push("json");
    if (formats.csv) selected.push("csv");
    if (formats.xlsx) selected.push("xlsx");
    if (formats.xml) selected.push("xml");
    return selected.join(",");
  }, [formats]);

  useEffect(() => {
    return () => {
      if (uploadCtrl) uploadCtrl.abort();
    };
  }, [uploadCtrl]);

  // =========================================================
  // HANDLER: Fáze 1 – Feature Discovery
  // =========================================================
  async function handleDiscover(sampleFile: File, labelsFile?: File | null) {
    setIsDiscovering(true);
    setFeatureSpec(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", sampleFile);
    formData.append("target_variable", targetVariable);
    formData.append("model", modelProvider);
    if (labelsFile) {
      formData.append("labels_file", labelsFile);
    }

    try {
      const res = await fetch(DISCOVER_URL, { method: "POST", body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "Feature Discovery selhala");
      }
      const data = await res.json();
      setFeatureSpec(data.suggested_features);
    } catch (err: unknown) {
      setError("Fáze 1 selhala: " + (err instanceof Error ? err.message : String(err)));
    } finally {
      setIsDiscovering(false);
    }
  }

  // =========================================================
  // HANDLER: Fáze 2 – Feature Extraction (training data)
  // =========================================================
  async function handleExtractTraining(zipFile: File, labelsFile?: File | null) {
    setExtractionBusy(true);
    setTrainingDataX(null);
    setProgress(0);
    setProgressLabel("Odesílám dataset...");
    setError(null);

    const formData = new FormData();
    formData.append("file", zipFile);
    formData.append("model", modelProvider);
    formData.append("feature_spec", JSON.stringify(featureSpec));
    formData.append("dataset_type", "training");
    if (labelsFile) {
      formData.append("labels_file", labelsFile);
    }

    try {
      const res = await fetch(EXTRACT_URL, { method: "POST", body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "Extrakce selhala");
      }
      const data = await res.json();

      if (data.job_id) {
        const ctrl = new AbortController();
        await pollProgress(
          data.job_id,
          (s: StatusPayload) => {
            setProgress(Math.max(0, Math.min(100, s.progress ?? 0)));
            setProgressLabel(s.stage || "");

            if (s.done && s.details?.status === "success") {
              setTrainingDataX(s.details.dataset_X);
              setDatasetYColumns(s.details.dataset_Y_columns || null);
            }
            if (s.done && s.error) {
              setError("Fáze 2 selhala: " + s.error);
            }
          },
          ctrl.signal,
        );
      }
    } catch (e: unknown) {
      setError("Fáze 2 selhala: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setExtractionBusy(false);
    }
  }

  // =========================================================
  // HANDLER: Fáze 3 – ML Training (RuleKit)
  // =========================================================
  async function handleTrain(targetColumn: string) {
    setTrainingBusy(true);
    setTrainResult(null);
    setError(null);

    try {
      const res = await fetch(TRAIN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_column: targetColumn }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "Trénink selhal");
      }
      const data: TrainResult = await res.json();
      setTrainResult(data);
    } catch (e: unknown) {
      setError("Fáze 3 selhala: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setTrainingBusy(false);
    }
  }

  // =========================================================
  // HANDLER: Fáze 4 – Test Data Feature Extraction
  // =========================================================
  async function handleExtractTesting(zipFile: File) {
    setTestExtractionBusy(true);
    setTestingDataX(null);
    setProgress(0);
    setProgressLabel("Odesílám testovací dataset...");
    setError(null);

    const formData = new FormData();
    formData.append("file", zipFile);
    formData.append("model", modelProvider);
    formData.append("feature_spec", JSON.stringify(featureSpec));
    formData.append("dataset_type", "testing");

    try {
      const res = await fetch(EXTRACT_URL, { method: "POST", body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "Extrakce selhala");
      }
      const data = await res.json();

      if (data.job_id) {
        const ctrl = new AbortController();
        await pollProgress(
          data.job_id,
          (s: StatusPayload) => {
            setProgress(Math.max(0, Math.min(100, s.progress ?? 0)));
            setProgressLabel(s.stage || "");

            if (s.done && s.details?.status === "success") {
              setTestingDataX(s.details.dataset_X);
            }
            if (s.done && s.error) {
              setError("Fáze 4 selhala: " + s.error);
            }
          },
          ctrl.signal,
        );
      }
    } catch (e: unknown) {
      setError("Fáze 4 selhala: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setTestExtractionBusy(false);
    }
  }

  // =========================================================
  // HANDLER: Fáze 5 – Predikce
  // =========================================================
  async function handlePredict(labelsFile?: File | null) {
    setPredictBusy(true);
    setPredictions(null);
    setPredictionMetrics(null);
    setError(null);

    try {
      let res: Response;
      if (labelsFile) {
        const formData = new FormData();
        formData.append("labels_file", labelsFile);
        res = await fetch(PREDICT_URL, { method: "POST", body: formData });
      } else {
        res = await fetch(PREDICT_URL, { method: "POST" });
      }
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || "Predikce selhala");
      }
      const data = await res.json();
      setPredictions(data.predictions);
      setPredictionMetrics(data.metrics || null);
    } catch (e: unknown) {
      setError("Fáze 5 selhala: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setPredictBusy(false);
    }
  }

  // =========================================================
  // HANDLER: Upload / Predict
  // =========================================================
  async function handleUpload() {
    if (!files.length) return setError("Nejdřív nahraj soubory.");
    if (!fileType)
      return setError("Detekce typu selhala – smíchal jsi různé přípony?");

    if (uploadCtrl) uploadCtrl.abort();

    const ctrl = new AbortController();
    setUploadCtrl(ctrl);

    setBusy(true);
    setError(null);
    setResponse(null);
    setStep(2);

    setHasRealProgress(false);
    setProgress(0);
    setProgressLabel("");

    try {
      const form = new FormData();
      for (const f of files) form.append("files", f, f.name);
      if (outputFormatsString) form.append("output_formats", outputFormatsString);
      if (description.trim()) form.append("description", description.trim());

      form.append("model", modelProvider);
      form.append("file_type", fileType);
      form.append("categories", categories);

      const res = await fetch(PREDICT_URL, {
        method: "POST",
        body: form,
        signal: ctrl.signal,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `${res.status} ${res.statusText}`);
      }

      const jsonRaw = await res.json();

      // Predict mód – odpověď z /predict
      if (jsonRaw.predictions) {
        setPredictionResult({
          score: jsonRaw.predictions[0]?.predicted_score ?? 0,
          rule_applied: jsonRaw.predictions[0]?.rule_applied ?? "",
          extracted_features: jsonRaw.predictions[0]?.extracted_features ?? {},
          testing_data_X: jsonRaw.predictions[0]?.extracted_features ?? {},
        });
      }
      setResponse(null);
      setStep(3);
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== "AbortError") {
        setError(e.message || "Nahrání selhalo");
        setStep(1);
      }
      setProgress(0);
      setProgressLabel("");
      setHasRealProgress(false);
    } finally {
      setBusy(false);
      setUploadCtrl(null);
    }
  }

  function handleRemove(index: number) {
    setFiles((prev) => {
      const next = prev.filter((_, i) => i !== index);
      if (next.length === 0) setFileType(null);
      return next;
    });
  }

  function handleReset() {
    if (uploadCtrl) uploadCtrl.abort();
    setUploadCtrl(null);
    setBusy(false);
    setFiles([]);
    setFileType(null);
    setError(null);
    setResponse(null);
    setPredictionResult(null);
    setCategories("");
    setProgress(0);
    setProgressLabel("");
    setHasRealProgress(false);
    setStep(1);
    // Reset training state
    setTrainingStep(1);
    setTargetVariable("movie memorability score");
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
    try {
      localStorage.removeItem("mflFilesMeta");
      localStorage.removeItem("mflFileType");
    } catch {
      /* localStorage unavailable */
    }
  }

  const outputs: Record<string, unknown> = useMemo(
    () => getOutputs(response?.processing),
    [response]
  );
  const procType = useMemo(
    () => getProcType(response?.processing),
    [response]
  );
  const transcript = useMemo(
    () => getTranscript(response?.processing),
    [response]
  );

  const [previews, setPreviews] = useState<string[]>([]);
  useEffect(() => {
    previews.forEach((u) => URL.revokeObjectURL(u));
    const next = files
      .filter(
        (f) => f.type.startsWith("image/") || f.type.startsWith("video/")
      )
      .map((f) => URL.createObjectURL(f));
    setPreviews(next);
    return () => next.forEach((u) => URL.revokeObjectURL(u));
  }, [files]);

  const defaultTab = useMemo<
    "overview" | "details" | "downloads" | "image" | "video"
  >(
    () => (Object.keys(outputs).length ? "downloads" : "overview"),
    [outputs]
  );

  // =========================================================
  // RENDER
  // =========================================================
  return (
    <div
      className={`min-h-screen ${
        deluxe
          ? "bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-white"
          : "bg-slate-50 text-slate-900"
      }`}
    >
      <div className="mx-auto max-w-5xl px-6 py-6">
        {/* HEADER */}
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <a
              href="https://www.vse.cz/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                src="/VSE_logo_CZ_circle_blue.png"
                alt="Logo školy"
                className="h-12 w-12 rounded-full shadow"
              />
            </a>
            <div>
              <h1
                className={`text-[28px] font-semibold tracking-tight ${
                  deluxe ? "text-white" : "text-slate-900"
                }`}
              >
                Media Feature Lab — Pro
              </h1>
              <p
                className={`mt-0.5 text-sm ${
                  deluxe ? "text-slate-300" : "text-slate-600"
                }`}
              >
                Prague University of Economics and Business
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div
              className={`p-1 rounded-lg flex ${
                deluxe ? "bg-white/10" : "bg-slate-200"
              }`}
            >
              <button
                onClick={() => setAppMode("predict")}
                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${
                  appMode === "predict"
                    ? deluxe
                      ? "bg-slate-700 text-white shadow"
                      : "bg-white text-slate-900 shadow"
                    : deluxe
                    ? "text-slate-400 hover:text-white"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Predikce
              </button>
              <button
                onClick={() => setAppMode("train")}
                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all ${
                  appMode === "train"
                    ? deluxe
                      ? "bg-slate-700 text-white shadow"
                      : "bg-white text-slate-900 shadow"
                    : deluxe
                    ? "text-slate-400 hover:text-white"
                    : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Trénink
              </button>
            </div>

            <Button
              variant={deluxe ? "secondary" : "outline"}
              size="icon"
              className="rounded-full"
              onClick={() => setDeluxe((v) => !v)}
              title="Přepnout motiv"
            >
              {deluxe ? (
                <Sun className="h-4 w-4" />
              ) : (
                <Moon className="h-4 w-4" />
              )}
            </Button>

            <Button
              variant={deluxe ? "secondary" : "default"}
              size="icon"
              className="rounded-full"
              onClick={() => setShowGuide(true)}
              title="Průvodce"
            >
              <HelpCircle className="h-4 w-4" />
            </Button>

            <Button variant="outline" onClick={handleReset}>
              <RefreshCw className="mr-2 h-4 w-4" /> Reset
            </Button>
          </div>
        </div>

        {/* BODY */}
        {appMode === "train" ? (
          <TrainingView
            deluxe={deluxe}
            /* Phase 1 */
            onDiscoverStart={handleDiscover}
            isDiscovering={isDiscovering}
            targetVariable={targetVariable}
            setTargetVariable={setTargetVariable}
            featureSpec={featureSpec}
            setFeatureSpec={setFeatureSpec}
            /* Phase 2 */
            onExtractTraining={handleExtractTraining}
            isExtracting={extractionBusy}
            trainingDataX={trainingDataX}
            datasetYColumns={datasetYColumns}
            /* Phase 3 */
            onTrain={handleTrain}
            isTraining={trainingBusy}
            trainResult={trainResult}
            /* Phase 4 */
            onExtractTesting={handleExtractTesting}
            isExtractingTest={testExtractionBusy}
            testingDataX={testingDataX}
            /* Phase 5 */
            onPredict={handlePredict}
            isPredicting={predictBusy}
            predictions={predictions}
            predictionMetrics={predictionMetrics}
            /* Common */
            modelProvider={modelProvider}
            setModelProvider={setModelProvider}
            step={trainingStep}
            onGoToStep={(s) => setTrainingStep(s as 1 | 2 | 3 | 4 | 5)}
            progress={progress}
            progressLabel={progressLabel}
            error={error}
            clearError={() => setError(null)}
          />
        ) : (
          <div className="grid grid-cols-1 gap-6">
            <Card
              className={`${
                deluxe
                  ? "bg-white/5 border-white/10"
                  : "bg-white border-slate-200"
              } backdrop-blur-xl`}
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle
                      className={`${
                        deluxe ? "text-white" : "text-slate-900"
                      } text-xl`}
                    >
                      Vstupní média
                    </CardTitle>
                    <CardDescription
                      className={`${
                        deluxe ? "text-slate-300" : "text-slate-600"
                      }`}
                    >
                      Nahraj více souborů stejného typu. Backend nepovoluje mix
                      typů v jednom requestu.
                    </CardDescription>
                  </div>
                  {fileType && (
                    <div
                      className={`px-2 py-1 rounded-lg text-xs font-medium ${TYPE_STYLES[fileType].chip} border ${TYPE_STYLES[fileType].border}`}
                    >
                      {TYPE_STYLES[fileType].label}
                    </div>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <motion.div
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                >
                  <div
                    {...getRootProps()}
                    className={`relative cursor-pointer rounded-2xl border-2 border-dashed p-8 transition ${
                      isDragActive
                        ? deluxe
                          ? "border-fuchsia-400/60 bg-fuchsia-500/10"
                          : "border-indigo-500 bg-indigo-50"
                        : deluxe
                        ? "border-white/10 hover:bg-white/5"
                        : "border-slate-300 hover:bg-slate-50"
                    } ${uploaderBorder(fileType)}`}
                  >
                    <input {...getInputProps()} />
                    <div className="flex flex-col items-center gap-3">
                      <div
                        className={`rounded-2xl ${
                          deluxe ? "bg-white/10" : "bg-white"
                        } p-4 shadow-sm`}
                      >
                        <UploadCloud
                          className={`${
                            deluxe ? "text-white" : "text-slate-800"
                          } h-9 w-9`}
                        />
                      </div>
                      <p
                        className={`text-[15px] ${
                          deluxe ? "text-slate-200" : "text-slate-700"
                        } font-medium`}
                      >
                        Přetáhni soubory nebo klikni pro výběr
                      </p>
                      <p
                        className={`text-xs ${
                          deluxe ? "text-slate-400" : "text-slate-500"
                        }`}
                      >
                        PDF / JPG / PNG / WEBP / MP4 / MOV / AVI / MKV / ZIP
                      </p>
                    </div>
                  </div>
                </motion.div>

                {error && (
                  <div
                    className={`mt-4 rounded-xl p-3 text-sm ${
                      deluxe
                        ? "bg-red-400/10 text-red-200"
                        : "bg-red-50 text-red-700"
                    }`}
                  >
                    {error}
                  </div>
                )}

                {files.length > 0 && (
                  <div className="mt-4">
                    <div
                      className={`mb-2 text-xs ${
                        deluxe ? "text-slate-300" : "text-slate-600"
                      }`}
                    >
                      Vybráno {files.length} souborů
                    </div>
                    <div className="flex flex-wrap gap-3">
                      {files.map((f, i) => {
                        const pal = tileBg(fileType);
                        return (
                          <div
                            key={fileKey(f)}
                            className={`group flex items-center gap-3 rounded-2xl border px-4 py-3 shadow-sm ${
                              fileType
                                ? TYPE_STYLES[fileType].border
                                : "border-slate-300"
                            } ${deluxe ? "bg-white/5" : "bg-white"}`}
                            style={{ minWidth: 300 }}
                          >
                            <span
                              className="inline-block h-10 w-1.5 rounded-full"
                              style={{ backgroundColor: pal.fg }}
                            />
                            <div
                              className="flex items-center gap-2 rounded-xl px-3 py-2"
                              style={{
                                background: pal.bg,
                                color: pal.fg,
                              }}
                            >
                              <span className="opacity-90">
                                {fileIcon(f.name)}
                              </span>
                              <div className="flex flex-col">
                                <span className="text-[13px] font-medium truncate max-w-[30ch]">
                                  {f.name}
                                </span>
                                <span className="text-[11px] opacity-80">
                                  {humanSize(f.size)}
                                </span>
                              </div>
                            </div>
                            <button
                              className={`ml-auto rounded-md p-1 ${
                                deluxe
                                  ? "hover:bg-white/10"
                                  : "hover:bg-slate-100"
                              }`}
                              onClick={() => handleRemove(i)}
                              title="Odebrat"
                            >
                              <X className="h-4 w-4" />
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
                  <div className="space-y-2 md:col-span-2">
                    <Label>Popis / Kontext</Label>
                    <textarea
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      rows={3}
                      className={`w-full rounded-xl border p-3 text-[13px] ${
                        deluxe
                          ? "bg-transparent border-white/20 text-white placeholder:text-slate-500"
                          : "bg-white border-slate-300 text-slate-900 placeholder:text-slate-400"
                      }`}
                      placeholder="Kontext datasetu, co má LLM zohlednit…"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label className={deluxe ? "text-slate-200" : ""}>
                      Formáty výstupu
                    </Label>
                    <div className="flex flex-wrap gap-3 text-sm">
                      {(["json", "csv", "xlsx", "xml"] as const).map((k) => (
                        <label
                          key={k}
                          className={`inline-flex items-center gap-2 rounded-xl px-2 py-1 ${
                            deluxe ? "bg-white/5" : "bg-slate-100"
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={formats[k]}
                            onChange={(e) =>
                              setFormats((s) => ({
                                ...s,
                                [k]: e.target.checked,
                              }))
                            }
                          />
                          {k.toUpperCase()}
                        </label>
                      ))}
                    </div>
                    <p
                      className={`text-xs ${
                        deluxe ? "text-slate-400" : "text-slate-500"
                      }`}
                    >
                      Backend očekává <code>output_formats</code> jako
                      comma-separated string.
                    </p>
                  </div>
                </div>

                {(busy || hasRealProgress) && (
                  <div className="mt-6">
                    {hasRealProgress ? (
                      <>
                        <div className="relative">
                          <Progress value={progress} />
                          <div
                            className={`pointer-events-none absolute -top-6 right-0 rounded-md px-2 py-0.5 text-xs font-medium ${
                              deluxe
                                ? "bg-white/10 text-white"
                                : "bg-black/10 text-slate-800"
                            } backdrop-blur`}
                          >
                            {Math.round(progress)}%
                          </div>
                        </div>
                        {progressLabel && (
                          <p
                            className={`mt-1 text-xs ${
                              deluxe ? "text-slate-300" : "text-slate-600"
                            }`}
                          >
                            {progressLabel}
                          </p>
                        )}
                      </>
                    ) : (
                      <div
                        className={`flex items-center gap-2 text-sm ${
                          deluxe ? "text-slate-300" : "text-slate-600"
                        }`}
                      />
                    )}
                  </div>
                )}

                <div className="mt-6 pt-4 border-t border-slate-100 dark:border-white/5 flex flex-wrap items-center justify-between gap-4">
                  <div className="flex items-center flex-wrap gap-3">
                    <Button
                      onClick={handleUpload}
                      disabled={!files.length || !!error || busy}
                      className="rounded-xl"
                    >
                      {busy ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />{" "}
                          Zpracovávám…
                        </>
                      ) : (
                        <>
                          <PlayCircle className="mr-2 h-4 w-4" /> Spustit
                          extrakci
                        </>
                      )}
                    </Button>

                    <div
                      className={`flex items-center gap-2 text-xs ${
                        deluxe ? "text-slate-300" : "text-slate-500"
                      }`}
                    >
                      <span
                        className={step >= 1 ? "font-medium" : "opacity-60"}
                      >
                        Krok 1: Nahrát
                      </span>
                      <span>→</span>
                      <span
                        className={step >= 2 ? "font-medium" : "opacity-60"}
                      >
                        Krok 2: Extrakce
                      </span>
                      <span>→</span>
                      <span
                        className={step >= 3 ? "font-medium" : "opacity-60"}
                      >
                        Krok 3: Export
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <span
                      className={`flex items-center gap-1.5 text-xs font-medium ${
                        deluxe ? "text-slate-400" : "text-slate-500"
                      }`}
                    >
                      <Cpu className="h-3.5 w-3.5" />
                      Zpracovat pomocí:
                    </span>
                    <div className="relative group">
                      <select
                        id="model_provider"
                        value={modelProvider}
                        onChange={(e) => setModelProvider(e.target.value)}
                        className={`appearance-none text-xs font-bold pl-2.5 pr-7 py-1.5 rounded-md outline-none transition-all cursor-pointer ${
                          deluxe
                            ? "bg-transparent text-white hover:bg-white/10"
                            : "bg-transparent text-slate-900 hover:bg-slate-100"
                        }`}
                        title="Výběr LLM modelu"
                      >
                        {AVAILABLE_MODELS.map((model) => (
                          <option
                            key={model.id}
                            value={model.id}
                            className={
                              deluxe
                                ? "bg-slate-800 text-white font-normal"
                                : "bg-white font-normal"
                            }
                          >
                            {model.name}
                          </option>
                        ))}
                      </select>
                      <ChevronDown
                        className={`absolute right-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 pointer-events-none ${
                          deluxe ? "text-slate-300" : "text-slate-600"
                        }`}
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* RESULTS */}
        {predictionResult && appMode === "predict" && (
          <div className="mt-6">
            <Card
              className={`${
                deluxe
                  ? "bg-white/5 border-white/10"
                  : "bg-white border-slate-200"
              } backdrop-blur-xl`}
            >
              <CardHeader>
                <CardTitle
                  className={deluxe ? "text-white" : "text-slate-900"}
                >
                  🎯 Výsledek Predikce
                </CardTitle>
                <CardDescription
                  className={deluxe ? "text-slate-300" : "text-slate-600"}
                >
                  Fáze 5: Predikce z ML modelu
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Fáze 5c: Predikovaný score */}
                <div
                  className={`rounded-xl p-4 ${
                    deluxe
                      ? "bg-green-900/30 border border-green-700/50"
                      : "bg-green-50 border border-green-200"
                  }`}
                >
                  <div
                    className={`text-sm space-y-2 ${
                      deluxe ? "text-slate-200" : "text-slate-700"
                    }`}
                  >
                    <div>
                      <span className="opacity-70">Predikovaná hodnota:</span>{" "}
                      <b className="text-lg text-green-600">{predictionResult.score.toFixed(4)}</b>
                    </div>
                  </div>
                </div>

                {/* Fáze 5c: Použité pravidlo */}
                <div
                  className={`rounded-xl p-4 ${
                    deluxe
                      ? "bg-blue-900/30 border border-blue-700/50"
                      : "bg-blue-50 border border-blue-200"
                  }`}
                >
                  <p
                    className={`text-xs font-bold mb-2 ${
                      deluxe ? "text-blue-300" : "text-blue-900"
                    }`}
                  >
                    📋 Použité pravidlo:
                  </p>
                  <p
                    className={`text-xs font-mono ${
                      deluxe ? "text-slate-300" : "text-slate-600"
                    }`}
                  >
                    {predictionResult.rule_applied}
                  </p>
                </div>

                {/* Extracted features */}
                <div
                  className={`rounded-xl p-4 ${
                    deluxe
                      ? "bg-slate-800/50 border border-slate-700/50"
                      : "bg-slate-50 border border-slate-200"
                  }`}
                >
                  <p
                    className={`text-xs font-bold mb-2 ${
                      deluxe ? "text-slate-300" : "text-slate-900"
                    }`}
                  >
                    🔍 Extrahované vlastnosti:
                  </p>
                  <div
                    className={`text-xs space-y-1 font-mono ${
                      deluxe ? "text-slate-400" : "text-slate-600"
                    }`}
                  >
                    {Object.entries(predictionResult.extracted_features).map(
                      ([key, val]) => (
                        <div key={key}>
                          <span className="opacity-60">{key}:</span>{" "}
                          <span className="font-bold">{String(val)}</span>
                        </div>
                      )
                    )}
                  </div>
                </div>

                {/* Download button */}
                <button
                  onClick={() =>
                    downloadTestingDataWithPrediction(
                      predictionResult.testing_data_X,
                      predictionResult.score,
                      predictionResult.rule_applied
                    )
                  }
                  className="w-full mt-4 px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 text-sm font-medium flex items-center justify-center gap-2"
                >
                  <Download className="w-4 h-4" /> Stáhnout výsledky
                </button>
              </CardContent>
            </Card>
          </div>
        )}

        {response && appMode === "predict" && (
          <div className="mt-6">
            <Card
              className={`${
                deluxe
                  ? "bg-white/5 border-white/10"
                  : "bg-white border-slate-200"
              } backdrop-blur-xl`}
            >
              <CardHeader>
                <CardTitle
                  className={deluxe ? "text-white" : "text-slate-900"}
                >
                  Výsledky z backendu
                </CardTitle>
                <CardDescription
                  className={deluxe ? "text-slate-300" : "text-slate-600"}
                >
                  {response.message}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue={defaultTab}>
                  <TabsList className={deluxe ? "bg-white/10" : ""}>
                    <TabsTrigger value="overview">Overview</TabsTrigger>
                    <TabsTrigger value="details">Processing JSON</TabsTrigger>
                    <TabsTrigger value="downloads">Downloads</TabsTrigger>
                    {procType === "image" && (
                      <TabsTrigger value="image">Image Features</TabsTrigger>
                    )}
                    {procType === "video" && (
                      <TabsTrigger value="video">Video Transcript</TabsTrigger>
                    )}
                  </TabsList>

                  <TabsContent value="overview" className="mt-4 space-y-4">
                    <div
                      className={`rounded-xl p-4 ${
                        deluxe
                          ? "bg-white/5 border border-white/10"
                          : "bg-slate-50"
                      }`}
                    >
                      <div
                        className={`text-sm ${
                          deluxe ? "text-slate-200" : "text-slate-700"
                        }`}
                      >
                        <div>
                          <span className="opacity-70">Uložené soubory:</span>{" "}
                          {response.files?.join(", ")}
                        </div>
                        <div className="mt-1">
                          <span className="opacity-70">Typ zpracování:</span>{" "}
                          <b className="capitalize">{procType || "?"}</b>
                        </div>
                        {response.processing?.description && (
                          <div className="mt-1">
                            <span className="opacity-70">Popis:</span>{" "}
                            {String(response.processing.description)}
                          </div>
                        )}
                        {response.processing?.status && (
                          <div className="mt-1">
                            <span className="opacity-70">Status:</span>{" "}
                            {String(response.processing.status)}
                          </div>
                        )}
                      </div>
                    </div>

                    {previews.length > 0 && (
                      <div>
                        <p
                          className={`mb-2 text-sm ${
                            deluxe ? "text-slate-300" : "text-slate-700"
                          }`}
                        >
                          Náhledy
                        </p>
                        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
                          {previews.map((src, idx) =>
                            files[idx]?.type.startsWith("video/") ? (
                              <video
                                key={idx}
                                src={src}
                                className="h-28 w-full rounded-xl object-cover"
                                muted
                                loop
                                autoPlay
                              />
                            ) : (
                              <img
                                key={idx}
                                src={src}
                                className="h-28 w-full rounded-xl object-cover"
                              />
                            )
                          )}
                        </div>
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="details" className="mt-4">
                    <div className="mb-2 flex items-center justify-between">
                      <span
                        className={`text-sm ${
                          deluxe ? "text-slate-300" : "text-slate-700"
                        }`}
                      >
                        Processing JSON
                      </span>
                      <CopyButton
                        getText={() =>
                          JSON.stringify(response.processing, null, 2)
                        }
                      />
                    </div>
                    <pre
                      className={`max-h-[420px] overflow-auto rounded-xl p-4 text-xs ${
                        deluxe
                          ? "bg-black/30 text-slate-100"
                          : "bg-slate-100 text-slate-900"
                      }`}
                    >
                      {JSON.stringify(response.processing, null, 2)}
                    </pre>
                  </TabsContent>

                  <TabsContent value="downloads" className="mt-4">
                    {Object.keys(outputs).length === 0 ? (
                      <p
                        className={`${
                          deluxe ? "text-slate-300" : "text-slate-600"
                        } text-sm`}
                      >
                        Žádné soubory k dispozici.
                      </p>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(outputs).map(([k, v]) => (
                          <Button
                            key={k}
                            className={`rounded-xl ${
                              deluxe
                                ? "bg-white/10 text-white hover:bg-white/20"
                                : ""
                            }`}
                            onClick={() => {
                              if (k === "json") {
                                const text =
                                  typeof v === "string"
                                    ? v
                                    : JSON.stringify(v, null, 2);
                                downloadText(
                                  "output.json",
                                  text,
                                  "application/json;charset=utf-8"
                                );
                              } else if (k === "csv") {
                                downloadText(
                                  "output.csv",
                                  String(v),
                                  "text/csv;charset=utf-8"
                                );
                              } else if (k === "xlsx") {
                                downloadXLSX(v, "output.xlsx");
                              } else if (k === "xml") {
                                downloadText(
                                  "output.xml",
                                  String(v),
                                  "application/xml;charset=utf-8"
                                );
                              } else if (k === "srt") {
                                downloadText(
                                  "transcript.srt",
                                  String(v),
                                  "text/plain;charset=utf-8"
                                );
                              } else if (k === "vtt") {
                                downloadText(
                                  "transcript.vtt",
                                  String(v),
                                  "text/vtt;charset=utf-8"
                                );
                              } else {
                                downloadText(`${k}.txt`, String(v));
                              }
                            }}
                          >
                            <Download className="mr-2 h-4 w-4" />{" "}
                            {k.toUpperCase()}
                          </Button>
                        ))}
                      </div>
                    )}
                  </TabsContent>

                  {procType === "image" && (
                    <TabsContent value="image" className="mt-4 space-y-4">
                      {response.processing?.feature_specification ? (
                        <div>
                          <div className="mb-2 flex items-center justify-between">
                            <p
                              className={`text-sm ${
                                deluxe ? "text-slate-300" : "text-slate-700"
                              }`}
                            >
                              Feature specification (echo)
                            </p>
                            <CopyButton
                              getText={() =>
                                typeof response.processing
                                  .feature_specification === "string"
                                  ? response.processing.feature_specification
                                  : JSON.stringify(
                                      response.processing.feature_specification,
                                      null,
                                      2
                                    )
                              }
                            />
                          </div>
                          <pre
                            className={`max-h-[360px] overflow-auto rounded-xl p-4 text-xs ${
                              deluxe
                                ? "bg-black/30 text-slate-100"
                                : "bg-slate-100 text-slate-900"
                            }`}
                          >
                            {typeof response.processing
                              .feature_specification === "string"
                              ? response.processing.feature_specification
                              : JSON.stringify(
                                  response.processing.feature_specification,
                                  null,
                                  2
                                )}
                          </pre>
                        </div>
                      ) : (
                        <p
                          className={`${
                            deluxe ? "text-slate-300" : "text-slate-600"
                          } text-sm`}
                        >
                          Chybí feature_specification.
                        </p>
                      )}

                      {response.processing?.tabular_output ? (
                        <div>
                          <div className="mb-2 flex items-center justify-between">
                            <p
                              className={`text-sm ${
                                deluxe ? "text-slate-300" : "text-slate-700"
                              }`}
                            >
                              Tabular output
                            </p>
                            <CopyButton
                              getText={() =>
                                typeof response.processing
                                  .tabular_output === "string"
                                  ? (response.processing
                                      .tabular_output as string)
                                  : JSON.stringify(
                                      response.processing.tabular_output,
                                      null,
                                      2
                                    )
                              }
                            />
                          </div>
                          <pre
                            className={`max-h-[360px] overflow-auto rounded-xl p-4 text-xs ${
                              deluxe
                                ? "bg-black/30 text-slate-100"
                                : "bg-slate-100 text-slate-900"
                            }`}
                          >
                            {typeof response.processing.tabular_output ===
                            "string"
                              ? response.processing.tabular_output
                              : JSON.stringify(
                                  response.processing.tabular_output,
                                  null,
                                  2
                                )}
                          </pre>
                        </div>
                      ) : null}
                    </TabsContent>
                  )}

                  {procType === "video" && (
                    <TabsContent value="video" className="mt-4 space-y-4">
                      <div className="mb-2 flex items-center justify-between">
                        <p
                          className={`text-sm ${
                            deluxe ? "text-slate-300" : "text-slate-700"
                          }`}
                        >
                          ASR Transkript
                        </p>
                        <CopyButton
                          getText={() => String(transcript || "")}
                        />
                      </div>
                      {transcript ? (
                        <pre
                          className={`max-h-[360px] whitespace-pre-wrap overflow-auto rounded-xl p-4 text-xs ${
                            deluxe
                              ? "bg-black/30 text-slate-100"
                              : "bg-slate-100 text-slate-900"
                          }`}
                        >
                          {transcript}
                        </pre>
                      ) : (
                        <p
                          className={`${
                            deluxe ? "text-slate-300" : "text-slate-600"
                          } text-sm`}
                        >
                          Transkript nebyl vrácen.
                        </p>
                      )}
                    </TabsContent>
                  )}
                </Tabs>
              </CardContent>
            </Card>
          </div>
        )}

        {/* FOOTER */}
        <footer className="mt-10">
          <div
            className={`h-px w-full ${
              deluxe
                ? "bg-gradient-to-r from-fuchsia-500/0 via-fuchsia-500/40 to-fuchsia-500/0"
                : "bg-gradient-to-r from-indigo-500/0 via-indigo-500/50 to-indigo-500/0"
            }`}
          />
          <div className="mt-5 flex flex-col items-center gap-2 text-center">
            <a
              href="https://www.vse.cz/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                src="/VSE_logo_CZ_circle_blue.png"
                alt="Logo školy"
                className={`h-10 w-10 ${
                  deluxe ? "opacity-90" : "opacity-80"
                }`}
              />
            </a>
            <p
              className={`${
                deluxe ? "text-slate-300" : "text-slate-600"
              } text-sm`}
            >
              &copy; {new Date().getFullYear()} Prague University of Economics
              and Business · Team 2 – Media Feature Lab
            </p>
          </div>
        </footer>
      </div>

      <Guide open={showGuide} onClose={() => setShowGuide(false)} />
    </div>
  );
}
