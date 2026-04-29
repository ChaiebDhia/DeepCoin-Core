"use client";

import dynamic from "next/dynamic";
import { useMemo, useRef, useState } from "react";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { useTranslations } from "next-intl";
import {
  BookOpen,
  Camera,
  ChartNoAxesCombined,
  CheckCircle2,
  Coins,
  Database,
  FlaskConical,
  Loader2,
  Plus,
  RefreshCw,
  Save,
  Search,
  Trash2,
  Upload,
  WandSparkles,
  ImagePlus,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

import {
  browseKb,
  coinImageDisplayUrl,
  createAdminCoin,
  deleteAdminCoin,
  getAdminCoinStats,
  getAdminCoins,
  prefillAdminCoin,
  prefillAdminCoinFromImage,
  updateAdminCoin,
  uploadAdminCoinImage,
} from "@/lib/api";
import type {
  AdminCoinItem,
  AdminCoinListResponse,
  AdminCoinPrefillResponse,
  AdminCoinStatsResponse,
  AdminCoinUpsertPayload,
  KbBrowseResponse,
  KbTypeItem,
} from "@/types/api";

const CoinLayerMap = dynamic(
  () => import("./CoinLayerMap").then((m) => m.CoinLayerMap),
  {
    ssr: false,
    loading: () => (
      <div className="rounded-xl border p-4 text-sm" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
        Loading map...
      </div>
    ),
  },
);

interface CoinInventoryTabProps {
  sessionStatus: string;
}

type CoinFormState = AdminCoinUpsertPayload;
type TrainingFilter = "all" | "training" | "external";
type MapLayerFilter = "all" | "training" | "rag_only" | "user_added";
type BrowserScope = "curated" | "corpus";

function blankCoin(): CoinFormState {
  return {
    type_id: "",
    title: "",
    denomination: "",
    authority: "",
    region: "",
    mint: "",
    date_range: "",
    material: "",
    obverse: "",
    reverse: "",
    provenance: "",
    discoverer_name: "",
    source_name: "",
    source_url: "",
    source_type: "manual",
    cartography: "",
    latitude: null,
    longitude: null,
    in_training_set: false,
    ai_prefilled: false,
    ai_confidence: null,
    notes: "",
    gallery_images: [],
  };
}

function StatCard({ icon: Icon, label, value, sub }: { icon: LucideIcon; label: string; value: string; sub: string }) {
  return (
    <div className="rounded-xl border p-4" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ backgroundColor: "#d4a85322" }}>
          <Icon size={16} style={{ color: "var(--brand-gold)" }} />
        </div>
        <div>
          <p className="text-xl font-black tabular-nums" style={{ color: "var(--text-primary)" }}>{value}</p>
          <p className="text-[11px] font-semibold" style={{ color: "var(--text-secondary)" }}>{label}</p>
          <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{sub}</p>
        </div>
      </div>
    </div>
  );
}

function StepCard({
  icon: Icon,
  title,
  description,
  done,
  readyLabel,
  pendingLabel,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  done: boolean;
  readyLabel: string;
  pendingLabel: string;
}) {
  return (
    <div className="rounded-xl border p-3" style={{ borderColor: done ? "#22c55e88" : "var(--border)", backgroundColor: "var(--surface-2)" }}>
      <div className="flex items-center gap-2 mb-2">
        <Icon size={14} style={{ color: done ? "#22c55e" : "#94a3b8" }} />
        <span className="text-xs font-bold" style={{ color: done ? "#22c55e" : "var(--text-muted)" }}>{done ? readyLabel : pendingLabel}</span>
      </div>
      <p className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>{title}</p>
      <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>{description}</p>
    </div>
  );
}

export function CoinInventoryTab({ sessionStatus }: CoinInventoryTabProps) {
  const { status: authStatus } = useSession();
  const queryClient = useQueryClient();
  const tAdmin = useTranslations("AdminDashboard");

  const isPrivileged = sessionStatus === "authenticated" || authStatus === "authenticated";

  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [regionFilter, setRegionFilter] = useState("");
  const [sourceTypeFilter, setSourceTypeFilter] = useState("");
  const [trainingFilter, setTrainingFilter] = useState<TrainingFilter>("all");
  const [browserScope, setBrowserScope] = useState<BrowserScope>("curated");
  const [mapLayer, setMapLayer] = useState<MapLayerFilter>("all");

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<CoinFormState>(blankCoin());
  const [showAdvancedForm, setShowAdvancedForm] = useState(false);

  const [showAdvancedLookup, setShowAdvancedLookup] = useState(false);
  const [prefillTypeId, setPrefillTypeId] = useState("");
  const [prefillQuery, setPrefillQuery] = useState("");

  const [scanFile, setScanFile] = useState<File | null>(null);
  const [scanPreviewUrl, setScanPreviewUrl] = useState<string | null>(null);
  const [scanFileName, setScanFileName] = useState<string | null>(null);
  const scanInputRef = useRef<HTMLInputElement | null>(null);

  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState<string | null>(null);
  const [uploadCaption, setUploadCaption] = useState("");
  const [uploadSource, setUploadSource] = useState("");

  const [prefillWarnings, setPrefillWarnings] = useState<string[]>([]);
  const [lastMessage, setLastMessage] = useState<string | null>(null);

  const pageSize = 10;

  const getErrorMessage = (error: unknown, fallback: string): string => {
    if (error && typeof error === "object") {
      const candidate = error as { detail?: unknown; message?: unknown };
      if (typeof candidate.detail === "string" && candidate.detail.trim()) return candidate.detail;
      if (typeof candidate.message === "string" && candidate.message.trim()) return candidate.message;
    }
    return fallback;
  };

  const statsQuery = useQuery<AdminCoinStatsResponse>({
    queryKey: ["admin", "coins", "stats"],
    queryFn: getAdminCoinStats,
    enabled: isPrivileged,
    staleTime: 30_000,
    refetchInterval: 30_000,
  });

  const inventoryQuery = useQuery<AdminCoinListResponse>({
    queryKey: ["admin", "coins", page, search, regionFilter, sourceTypeFilter, trainingFilter],
    queryFn: () =>
      getAdminCoins(
        (page - 1) * pageSize,
        pageSize,
        search || undefined,
        regionFilter || undefined,
        sourceTypeFilter || undefined,
        trainingFilter === "training" ? true : trainingFilter === "external" ? false : undefined,
      ),
    enabled: isPrivileged,
    placeholderData: (prev) => prev,
    staleTime: 15_000,
  });

  const kbQuery = useQuery<KbBrowseResponse>({
    queryKey: ["admin", "coins", "kb", page, search, trainingFilter],
    queryFn: () => browseKb(search || "", (page - 1) * pageSize, pageSize, trainingFilter === "training"),
    enabled: isPrivileged && browserScope === "corpus",
    placeholderData: (prev) => prev,
    staleTime: 20_000,
  });

  const prefillMutation = useMutation({
    mutationFn: prefillAdminCoin,
    onSuccess: (data: AdminCoinPrefillResponse) => {
      setForm(data.coin);
      setPrefillWarnings(data.warnings ?? []);
      setLastMessage(data.duplicate_exists ? "Draft loaded. Similar coin already exists." : "Draft loaded from corpus search.");
    },
    onError: (error: unknown) => setLastMessage(getErrorMessage(error, "Prefill failed.")),
  });

  const prefillImageMutation = useMutation({
    mutationFn: async () => {
      if (!scanFile) throw new Error("Choose an image first.");
      return prefillAdminCoinFromImage(scanFile);
    },
    onSuccess: (data: AdminCoinPrefillResponse) => {
      setForm(data.coin);
      setPrefillWarnings(data.warnings ?? []);
      setLastMessage("AI draft generated from image. Review and save.");
      if (scanFile) setScanFileName(scanFile.name);
      setScanFile(null);
    },
    onError: (error: unknown) => setLastMessage(getErrorMessage(error, "Image scan failed.")),
  });

  const saveMutation = useMutation({
    mutationFn: async () => {
      if (selectedId) return updateAdminCoin(selectedId, form);
      return createAdminCoin(form);
    },
    onSuccess: (item) => {
      setSelectedId(item.id);
      setLastMessage(`Saved coin ${item.type_id}.`);
      queryClient.invalidateQueries({ queryKey: ["admin", "coins"] });
      queryClient.invalidateQueries({ queryKey: ["admin", "coins", "stats"] });
    },
    onError: (error: unknown) => setLastMessage(getErrorMessage(error, "Save failed.")),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteAdminCoin,
    onSuccess: () => {
      setSelectedId(null);
      setForm(blankCoin());
      setLastMessage("Coin deleted.");
      queryClient.invalidateQueries({ queryKey: ["admin", "coins"] });
      queryClient.invalidateQueries({ queryKey: ["admin", "coins", "stats"] });
    },
    onError: (error: unknown) => setLastMessage(getErrorMessage(error, "Delete failed.")),
  });

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!selectedId || !uploadFile) throw new Error("Select a coin and image first.");
      return uploadAdminCoinImage(selectedId, uploadFile, uploadCaption, uploadSource);
    },
    onSuccess: (item) => {
      setForm((curr) => ({ ...curr, gallery_images: item.gallery_images }));
      setUploadFile(null);
      if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
      setUploadPreviewUrl(null);
      setUploadCaption("");
      setUploadSource("");
      setLastMessage("Image uploaded to this coin.");
      queryClient.invalidateQueries({ queryKey: ["admin", "coins"] });
      queryClient.invalidateQueries({ queryKey: ["admin", "coins", "stats"] });
    },
    onError: (error: unknown) => setLastMessage(getErrorMessage(error, "Image upload failed.")),
  });

  const stats = statsQuery.data;
  const rows = inventoryQuery.data?.items ?? [];
  const totalPages = Math.max(1, inventoryQuery.data?.pages ?? 1);

  const coordinateError = (() => {
    if (form.latitude == null && form.longitude == null) return null;
    if (form.latitude == null || form.longitude == null) return "Latitude and longitude must both be set.";
    if (form.latitude < -90 || form.latitude > 90) return "Latitude must be between -90 and 90.";
    if (form.longitude < -180 || form.longitude > 180) return "Longitude must be between -180 and 180.";
    return null;
  })();

  const steps = {
    scanned: Boolean(scanFile || scanPreviewUrl || form.ai_prefilled),
    drafted: Boolean(form.title || form.denomination || form.material),
    ready: Boolean(form.title && form.denomination && !coordinateError),
  };

  const selectedPrimaryImage = useMemo(() => {
    if (!form.gallery_images?.length) return null;
    return form.gallery_images.find((img) => img.is_primary) ?? form.gallery_images[0];
  }, [form.gallery_images]);

  const loadCoin = (item: AdminCoinItem) => {
    setSelectedId(item.id);
    setForm(item);
    setPrefillWarnings([]);
    setUploadFile(null);
    if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    setUploadPreviewUrl(null);
    setUploadCaption("");
    setUploadSource("");
    setScanFile(null);
    setScanFileName(null);
    if (scanPreviewUrl) URL.revokeObjectURL(scanPreviewUrl);
    setScanPreviewUrl(null);
    setLastMessage(null);
  };

  const onPrefillLookup = () => {
    prefillMutation.mutate({
      type_id: prefillTypeId.trim() || undefined,
      query: prefillQuery.trim() || undefined,
    });
  };

  const loadKbCoin = (item: KbTypeItem) => {
    setSelectedId(null);
    setLastMessage("Loading corpus record into form...");
    prefillMutation.mutate({ type_id: item.type_id });
  };

  const updateField = <K extends keyof CoinFormState>(key: K, value: CoinFormState[K]) => {
    setForm((curr) => ({ ...curr, [key]: value }));
  };

  const resetForm = () => {
    setSelectedId(null);
    setForm(blankCoin());
    setLastMessage("Form reset.");
    if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
    setUploadPreviewUrl(null);
    setUploadFile(null);
    setScanFile(null);
    setScanFileName(null);
    if (scanPreviewUrl) URL.revokeObjectURL(scanPreviewUrl);
    setScanPreviewUrl(null);
  };

  if (!isPrivileged) {
    return (
      <div className="rounded-xl border p-5" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
        <p className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Admin or curator account required.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border p-5" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
        <div className="flex flex-col lg:flex-row gap-4 lg:items-center lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.2em]" style={{ color: "var(--text-muted)" }}>{tAdmin("workflow_title")}</p>
            <h3 className="text-xl font-black mt-1" style={{ color: "var(--text-primary)" }}>{tAdmin("workflow_subtitle")}</h3>
            <p className="text-sm mt-2" style={{ color: "var(--text-secondary)" }}>
              {tAdmin("scan_intro")}
            </p>
          </div>
          <button
            onClick={() => {
              if (!scanFile) {
                scanInputRef.current?.click();
                return;
              }
              prefillImageMutation.mutate();
            }}
            disabled={prefillImageMutation.isPending}
            className="rounded-xl px-4 py-3 text-sm font-bold inline-flex items-center gap-2 transition-transform hover:scale-[1.02] cursor-pointer disabled:opacity-60"
            style={{ backgroundColor: "#2563eb", color: "white" }}
          >
            {prefillImageMutation.isPending ? <Loader2 size={15} className="animate-spin" /> : <Camera size={15} />}
            {prefillImageMutation.isPending ? "Analyzing..." : scanFile ? tAdmin("analyze_selected") : tAdmin("step1_title")}
          </button>
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
          <StepCard icon={Camera} title={tAdmin("step1_title")} description={tAdmin("step1_desc")} done={steps.scanned} readyLabel={tAdmin("active")} pendingLabel={tAdmin("step1_pending")} />
          <StepCard icon={WandSparkles} title={tAdmin("step2_title")} description={tAdmin("step2_desc")} done={steps.drafted} readyLabel={tAdmin("active")} pendingLabel={tAdmin("step2_pending")} />
          <StepCard icon={Save} title={tAdmin("step3_title")} description={tAdmin("step3_desc")} done={steps.ready} readyLabel={tAdmin("active")} pendingLabel={tAdmin("step3_pending")} />
        </div>

        <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <label className="rounded-xl border p-4 block cursor-pointer hover:ring-1 transition" style={{ borderColor: "#3b82f655", backgroundColor: "var(--surface-2)", boxShadow: "inset 0 0 0 1px rgba(59,130,246,0.12)" }}>
            <p className="text-xs font-bold mb-2" style={{ color: "var(--text-secondary)" }}>{tAdmin("action_area")}</p>
            <input
              ref={scanInputRef}
              type="file"
              accept="image/*"
              className="text-sm w-full"
              onChange={(e) => {
                const file = e.target.files?.[0] ?? null;
                setScanFile(file);
                setScanFileName(file?.name ?? null);
                if (scanPreviewUrl) URL.revokeObjectURL(scanPreviewUrl);
                setScanPreviewUrl(file ? URL.createObjectURL(file) : null);
              }}
            />
            <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>
              {scanFile ? tAdmin("ready_to_analyze") : tAdmin("choose_image_click")}
            </p>
            {scanPreviewUrl ? (
              <div className="mt-2 rounded-lg border p-2" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
                {scanFileName ? <p className="text-[10px] mb-1 truncate" style={{ color: "var(--text-muted)" }}>{scanFileName}</p> : null}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={scanPreviewUrl} alt="Selected scan" className="h-44 w-full object-cover rounded" />
              </div>
            ) : null}
            {scanPreviewUrl && !scanFile ? (
              <p className="text-[10px] mt-2" style={{ color: "var(--text-muted)" }}>
                Draft generated from current image. Choose another image anytime to re-analyze.
              </p>
            ) : null}
          </label>

          <div className="rounded-xl border p-4" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }}>
            <div className="flex items-center justify-between">
              <p className="text-xs font-bold" style={{ color: "var(--text-secondary)" }}>{tAdmin("advanced_lookup")}</p>
              <button onClick={() => setShowAdvancedLookup((s) => !s)} className="text-xs font-semibold" style={{ color: "#93c5fd" }}>
                {showAdvancedLookup ? tAdmin("hide") : tAdmin("show")}
              </button>
            </div>
            {showAdvancedLookup ? (
              <div className="mt-2 space-y-2">
                <input value={prefillTypeId} onChange={(e) => setPrefillTypeId(e.target.value)} className="w-full rounded-lg border px-3 py-2 text-sm" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }} placeholder={tAdmin("known_corpus_id")} />
                <input value={prefillQuery} onChange={(e) => setPrefillQuery(e.target.value)} className="w-full rounded-lg border px-3 py-2 text-sm" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }} placeholder={tAdmin("keyword_lookup")} />
                <button onClick={onPrefillLookup} disabled={prefillMutation.isPending} className="rounded-lg px-3 py-2 text-xs font-bold inline-flex items-center gap-2 disabled:opacity-60" style={{ backgroundColor: "var(--brand-gold)", color: "#10131a" }}>
                  {prefillMutation.isPending ? <Loader2 size={12} className="animate-spin" /> : <WandSparkles size={12} />}
                  {tAdmin("prefill_draft")}
                </button>
              </div>
            ) : (
              <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>{tAdmin("usually_not_needed")}</p>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-4">
        <StatCard icon={Database} label={tAdmin("corpus")} value={(stats?.kb_total ?? 0).toLocaleString()} sub={tAdmin("types")} />
        <StatCard icon={BookOpen} label={tAdmin("inventory_total")} value={(stats?.total ?? 0).toLocaleString()} sub={tAdmin("all_curated")} />
        <StatCard icon={CheckCircle2} label={tAdmin("training")} value={(stats?.kb_training_total ?? 0).toLocaleString()} sub={tAdmin("cnn_classes")} />
        <StatCard icon={Coins} label={tAdmin("rag_only_types")} value={(stats?.kb_rag_only_total ?? 0).toLocaleString()} sub={tAdmin("knowledge_only")} />
        <StatCard icon={BookOpen} label={tAdmin("user_inventory")} value={(stats?.user_total ?? 0).toLocaleString()} sub={tAdmin("curated_only")} />
        <StatCard icon={FlaskConical} label={tAdmin("with_gallery")} value={(stats?.with_gallery ?? 0).toLocaleString()} sub={tAdmin("records")} />
      </div>

      <div className="rounded-xl border p-5" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
        <div className="flex items-center gap-2 mb-3">
          <ChartNoAxesCombined size={15} style={{ color: "var(--brand-gold)" }} />
          <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>{tAdmin("corpus_and_inventory")}</span>
        </div>

        <div className="flex flex-wrap gap-2 mb-3">
          {(["all", "training", "rag_only", "user_added"] as MapLayerFilter[]).map((l) => (
            <button
              key={l}
              onClick={() => setMapLayer(l)}
              className="text-[11px] rounded-full px-3 py-1.5 font-semibold border"
              style={{
                borderColor: mapLayer === l ? "#60a5fa" : "var(--border)",
                color: mapLayer === l ? "#dbeafe" : "var(--text-muted)",
                backgroundColor: mapLayer === l ? "#1e3a8a55" : "transparent",
              }}
            >
              {l === "all" ? tAdmin("all_layers") : tAdmin(l)}
            </button>
          ))}
        </div>

        <CoinLayerMap points={stats?.map_points ?? []} layer={mapLayer} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <div className="rounded-xl border overflow-hidden" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
          <div className="flex items-center gap-2 px-5 py-3.5 border-b" style={{ borderColor: "var(--border)" }}>
            <Search size={14} style={{ color: "var(--brand-gold)" }} />
            <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>{tAdmin("inventory_browser")}</span>
            <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ color: "#93c5fd", backgroundColor: "#3b82f622" }}>{tAdmin("global_scope")}</span>
            <span className="text-[10px] ml-2" style={{ color: "var(--text-muted)" }}>
              {browserScope === "curated" ? tAdmin("click_row") : "Click corpus row to load it into the form"}
            </span>
            <button onClick={() => queryClient.invalidateQueries({ queryKey: ["admin", "coins"] })} className="ml-auto text-xs inline-flex items-center gap-1.5 hover:underline" style={{ color: "var(--text-muted)" }}>
              <RefreshCw size={11} /> {tAdmin("refresh")}
            </button>
          </div>

          <div className="p-4 border-b space-y-3" style={{ borderColor: "var(--border)" }}>
            <div className="flex gap-2">
              <button
                onClick={() => { setBrowserScope("curated"); setPage(1); }}
                className="text-[11px] rounded-full px-3 py-1.5 font-semibold border"
                style={{
                  borderColor: browserScope === "curated" ? "#60a5fa" : "var(--border)",
                  color: browserScope === "curated" ? "#dbeafe" : "var(--text-muted)",
                  backgroundColor: browserScope === "curated" ? "#1e3a8a55" : "transparent",
                }}
              >
                {tAdmin("curated_user")}
              </button>
              <button
                onClick={() => { setBrowserScope("corpus"); setPage(1); }}
                className="text-[11px] rounded-full px-3 py-1.5 font-semibold border"
                style={{
                  borderColor: browserScope === "corpus" ? "#60a5fa" : "var(--border)",
                  color: browserScope === "corpus" ? "#dbeafe" : "var(--text-muted)",
                  backgroundColor: browserScope === "corpus" ? "#1e3a8a55" : "transparent",
                }}
              >
                {tAdmin("full_corpus")}
              </button>
            </div>
            <input value={search} onChange={(e) => { setSearch(e.target.value); setPage(1); }} className="w-full rounded-lg border px-3 py-2 text-sm" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }} placeholder={tAdmin("search_title_by")} />
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <input value={regionFilter} onChange={(e) => { setRegionFilter(e.target.value); setPage(1); }} className="rounded-lg border px-3 py-2 text-sm" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }} placeholder={tAdmin("region")} />
              <select value={sourceTypeFilter} onChange={(e) => { setSourceTypeFilter(e.target.value); setPage(1); }} className="rounded-lg border px-3 py-2 text-sm" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }}>
                <option value="">{tAdmin("all_source_types")}</option>
                <option value="manual">{tAdmin("manual")}</option>
                <option value="image-ai">{tAdmin("image_ai")}</option>
                <option value="kb">KB</option>
                <option value="search">Search</option>
                <option value="import">Import</option>
              </select>
              <select value={trainingFilter} onChange={(e) => { setTrainingFilter(e.target.value as TrainingFilter); setPage(1); }} className="rounded-lg border px-3 py-2 text-sm" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }}>
                <option value="all">{tAdmin("all_records")}</option>
                <option value="training">Training only</option>
                <option value="external">External only</option>
              </select>
            </div>
          </div>

          <div className="overflow-auto max-h-[560px]">
            <table className="w-full text-xs">
              <thead className="sticky top-0 z-10" style={{ backgroundColor: "var(--surface-1)" }}>
                <tr style={{ color: "var(--text-muted)" }}>
                  <th className="text-left px-4 py-3 font-semibold">{tAdmin("type")}</th>
                  <th className="text-left px-4 py-3 font-semibold">{tAdmin("coin")}</th>
                  <th className="text-left px-4 py-3 font-semibold">{tAdmin("source")}</th>
                  <th className="text-left px-4 py-3 font-semibold">{tAdmin("updated")}</th>
                </tr>
              </thead>
              <tbody>
                {browserScope === "curated" && inventoryQuery.isLoading ? (
                  <tr><td className="px-4 py-6" colSpan={4} style={{ color: "var(--text-muted)" }}>Loading...</td></tr>
                ) : browserScope === "corpus" && kbQuery.isLoading ? (
                  <tr><td className="px-4 py-6" colSpan={4} style={{ color: "var(--text-muted)" }}>Loading corpus...</td></tr>
                ) : browserScope === "curated" && rows.length ? rows.map((item) => (
                  <tr key={item.id} onClick={() => loadCoin(item)} className="cursor-pointer hover:bg-[rgba(212,168,83,0.08)]" style={{ backgroundColor: selectedId === item.id ? "rgba(212,168,83,0.1)" : "transparent" }}>
                    <td className="px-4 py-3 whitespace-nowrap font-semibold" style={{ color: "var(--text-primary)" }}>{item.type_id}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {item.gallery_images[0]?.url ? (
                          <>
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                              src={`${coinImageDisplayUrl(item.gallery_images[0].url)}?v=${encodeURIComponent(item.updated_at)}`}
                              alt={item.title ?? "coin image"}
                              className="w-8 h-8 rounded border object-cover"
                              style={{ borderColor: "var(--border)" }}
                              onError={(e) => {
                                e.currentTarget.style.display = "none";
                              }}
                            />
                          </>
                        ) : null}
                        <div>
                          <div className="font-bold" style={{ color: "var(--text-primary)" }}>{item.title}</div>
                          <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>{item.denomination ?? ""}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3" style={{ color: "var(--text-secondary)" }}>{item.source_type}</td>
                    <td className="px-4 py-3 tabular-nums" style={{ color: "var(--text-muted)" }}>{item.updated_at.slice(0, 10)}</td>
                  </tr>
                )) : browserScope === "corpus" && (kbQuery.data?.items?.length ?? 0) > 0 ? kbQuery.data!.items.map((item) => (
                  <tr key={`kb-${item.type_id}`} onClick={() => loadKbCoin(item)} className="cursor-pointer hover:bg-[rgba(59,130,246,0.08)]">
                    <td className="px-4 py-3 whitespace-nowrap font-semibold" style={{ color: "var(--text-primary)" }}>{item.type_id}</td>
                    <td className="px-4 py-3">
                      <div className="font-bold" style={{ color: "var(--text-primary)" }}>{item.denomination || "Unknown"}</div>
                      <div className="text-[10px]" style={{ color: "var(--text-muted)" }}>{item.mint || item.region || tAdmin("mint")}</div>
                    </td>
                    <td className="px-4 py-3" style={{ color: item.in_training_set ? "#22c55e" : "#60a5fa" }}>
                      {item.in_training_set ? "training" : "rag_only"}
                    </td>
                    <td className="px-4 py-3 tabular-nums" style={{ color: "var(--text-muted)" }}>{item.date_range || "-"}</td>
                  </tr>
                )) : (
                  <tr><td className="px-4 py-6" colSpan={4} style={{ color: "var(--text-muted)" }}>No records found.</td></tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between px-5 py-3 border-t" style={{ borderColor: "var(--border)" }}>
            <span className="text-xs" style={{ color: "var(--text-muted)" }}>
              {tAdmin("page")} {page} / {Math.max(1, browserScope === "curated" ? totalPages : Math.ceil((kbQuery.data?.total ?? 0) / pageSize) || 1)}
            </span>
            <div className="flex gap-2">
              <button disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))} className="px-2 py-1 rounded border disabled:opacity-40" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>{tAdmin("prev")}</button>
              <button
                disabled={page >= Math.max(1, browserScope === "curated" ? totalPages : Math.ceil((kbQuery.data?.total ?? 0) / pageSize) || 1)}
                onClick={() => setPage((p) => p + 1)}
                className="px-2 py-1 rounded border disabled:opacity-40"
                style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
              >
                {tAdmin("next")}
              </button>
            </div>
          </div>

          <div className="border-t p-4" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }}>
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs font-bold" style={{ color: "var(--text-secondary)" }}>{tAdmin("gallery_images")}</p>
              <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{selectedId ? `${form.gallery_images.length} image(s)` : tAdmin("select_coin")}</p>
            </div>
            {selectedId && form.gallery_images.length ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {form.gallery_images.map((image) => (
                  <div
                    key={`box-${image.filename}`}
                    className="rounded-lg border overflow-hidden"
                    style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`${coinImageDisplayUrl(image.url)}?v=${encodeURIComponent(image.filename)}`}
                      alt={image.caption ?? "coin image"}
                      className="h-24 w-full object-cover"
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                      }}
                    />
                    <div className="p-1.5 text-[10px] truncate" style={{ color: "var(--text-muted)" }}>
                      {image.caption || "Coin image"}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs" style={{ color: "var(--text-muted)" }}>{tAdmin("no_images")}</p>
            )}
          </div>
        </div>

        <div className="rounded-xl border p-5" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
          <div className="flex items-center gap-2 mb-4">
            <BookOpen size={15} style={{ color: "var(--brand-gold)" }} />
            <span className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>{selectedId ? `Editing ${form.type_id || tAdmin("coin")}` : tAdmin("curate_new_coin")}</span>
            {selectedId ? <span className="ml-auto text-[10px] rounded-full px-2 py-0.5" style={{ backgroundColor: "#3b82f622", color: "#93c5fd" }}>existing</span> : null}
          </div>

          {selectedPrimaryImage ? (
            <div className="mb-4 rounded-lg border overflow-hidden" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }}>
              <div className="px-3 py-2 border-b flex items-center justify-between" style={{ borderColor: "var(--border)" }}>
                <p className="text-xs font-bold" style={{ color: "var(--text-secondary)" }}>{tAdmin("coin")} Preview</p>
                <a href={selectedPrimaryImage.url} target="_blank" rel="noopener noreferrer" className="text-[10px] hover:underline" style={{ color: "#93c5fd" }}>
                  Open full image
                </a>
              </div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`${coinImageDisplayUrl(selectedPrimaryImage.url)}?v=${encodeURIComponent(selectedPrimaryImage.filename)}`}
                alt={selectedPrimaryImage.caption ?? "selected coin image"}
                className="h-52 w-full object-cover"
                onError={(e) => {
                  e.currentTarget.style.display = "none";
                }}
              />
            </div>
          ) : null}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Field label="Title" value={form.title ?? ""} onChange={(v) => updateField("title", v)} />
            <Field label="Denomination" value={form.denomination ?? ""} onChange={(v) => updateField("denomination", v)} />
            <Field label="Material" value={form.material ?? ""} onChange={(v) => updateField("material", v)} />
            <Field label="Region" value={form.region ?? ""} onChange={(v) => updateField("region", v)} />
            <Field label="Mint" value={form.mint ?? ""} onChange={(v) => updateField("mint", v)} />
            <Field label="Scientist Name (optional)" value={form.discoverer_name ?? ""} onChange={(v) => updateField("discoverer_name", v)} />
          </div>

          {coordinateError ? <p className="text-xs mt-2" style={{ color: "#ef4444" }}>{coordinateError}</p> : null}

          <div className="mt-3">
            <TextAreaField label="Notes (optional)" value={form.notes ?? ""} onChange={(v) => updateField("notes", v)} />
          </div>

          <div className="mt-3 flex items-center justify-between">
            <p className="text-xs font-bold" style={{ color: "var(--text-secondary)" }}>Advanced metadata</p>
            <button onClick={() => setShowAdvancedForm((s) => !s)} className="text-xs font-semibold" style={{ color: "#93c5fd" }}>
              {showAdvancedForm ? "Hide" : "Show"}
            </button>
          </div>

          {showAdvancedForm ? (
            <div className="mt-2 space-y-3">
              <Field label="Internal Type ID (optional)" value={form.type_id ?? ""} onChange={(v) => updateField("type_id", v)} disabled={Boolean(selectedId)} />
              <Field label="Authority" value={form.authority ?? ""} onChange={(v) => updateField("authority", v)} />
              <Field label="Date Range" value={form.date_range ?? ""} onChange={(v) => updateField("date_range", v)} />
              <Field label="Latitude" value={form.latitude ?? ""} onChange={(v) => updateField("latitude", v ? Number(v) : null)} />
              <Field label="Longitude" value={form.longitude ?? ""} onChange={(v) => updateField("longitude", v ? Number(v) : null)} />
              <Field label="Source Name" value={form.source_name ?? ""} onChange={(v) => updateField("source_name", v)} />
              <Field label="Source URL" value={form.source_url ?? ""} onChange={(v) => updateField("source_url", v)} />
              <TextAreaField label="Obverse" value={form.obverse ?? ""} onChange={(v) => updateField("obverse", v)} />
              <TextAreaField label="Reverse" value={form.reverse ?? ""} onChange={(v) => updateField("reverse", v)} />
            </div>
          ) : null}

          <div className="mt-4 rounded-lg border p-3" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }}>
            <p className="text-xs font-bold mb-2" style={{ color: "var(--text-secondary)" }}>Gallery images</p>
            <label
              className="group block rounded-lg border border-dashed p-3 cursor-pointer transition"
              style={{ borderColor: "#3b82f666", backgroundColor: "#2563eb11" }}
            >
              <div className="flex items-center gap-2 text-sm font-semibold" style={{ color: "#93c5fd" }}>
                <ImagePlus size={16} />
                Click to choose image
              </div>
              <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
                Supported: JPG, PNG, WEBP. This image is attached to the selected coin.
              </p>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0] ?? null;
                  setUploadFile(file);
                  if (uploadPreviewUrl) URL.revokeObjectURL(uploadPreviewUrl);
                  setUploadPreviewUrl(file ? URL.createObjectURL(file) : null);
                }}
              />
            </label>

            <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
              <input value={uploadCaption} onChange={(e) => setUploadCaption(e.target.value)} className="rounded-lg border px-3 py-2 text-sm" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }} placeholder="Caption (optional)" />
              <input value={uploadSource} onChange={(e) => setUploadSource(e.target.value)} className="rounded-lg border px-3 py-2 text-sm" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }} placeholder="Source (optional)" />
            </div>
            {uploadPreviewUrl ? (
              <div className="mt-2 rounded-lg border p-2" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}>
                <p className="text-[10px] mb-1" style={{ color: "var(--text-muted)" }}>Selected gallery preview</p>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={uploadPreviewUrl} alt="Selected upload" className="h-28 w-full object-cover rounded" />
              </div>
            ) : null}
            <div className="mt-2">
              <button
                disabled={!selectedId || !uploadFile || uploadMutation.isPending}
                onClick={() => uploadMutation.mutate()}
                className="rounded-lg px-3 py-2 text-xs font-bold inline-flex items-center gap-2 disabled:opacity-50 transition-transform hover:scale-[1.02]"
                style={{ backgroundColor: "#2563eb", color: "white" }}
              >
                {uploadMutation.isPending ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />}
                Upload to selected coin
              </button>
            </div>

            {form.gallery_images.length > 0 ? (
              <div className="mt-3 grid grid-cols-2 gap-2">
                {form.gallery_images.map((image) => (
                  <figure key={image.filename} className="rounded-lg border overflow-hidden" style={{ borderColor: "var(--border)" }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`${coinImageDisplayUrl(image.url)}?v=${encodeURIComponent(image.filename)}`}
                      alt={image.caption ?? "coin gallery image"}
                      className="h-28 w-full object-cover"
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                      }}
                    />
                    <figcaption className="p-2 text-[10px]" style={{ color: "var(--text-muted)" }}>{image.caption || "Coin image"}</figcaption>
                  </figure>
                ))}
              </div>
            ) : null}
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <button onClick={() => saveMutation.mutate()} disabled={saveMutation.isPending || Boolean(coordinateError)} className="rounded-lg px-4 py-2.5 text-sm font-bold inline-flex items-center gap-2 disabled:opacity-50" style={{ backgroundColor: "var(--brand-gold)", color: "#10131a" }}>
              {saveMutation.isPending ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
              {selectedId ? "Update coin" : "Create coin"}
            </button>
            <button onClick={resetForm} className="rounded-lg px-4 py-2.5 text-sm font-bold inline-flex items-center gap-2" style={{ backgroundColor: "var(--surface-2)", color: "var(--text-primary)", border: "1px solid var(--border)" }}>
              <Plus size={14} /> New form
            </button>
            <button onClick={() => { if (!selectedId) return; if (!window.confirm("Delete this coin record?")) return; deleteMutation.mutate(selectedId); }} disabled={!selectedId || deleteMutation.isPending} className="rounded-lg px-4 py-2.5 text-sm font-bold inline-flex items-center gap-2 disabled:opacity-50" style={{ backgroundColor: "#7f1d1d", color: "white" }}>
              {deleteMutation.isPending ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
              Delete
            </button>
          </div>

          {prefillWarnings.length > 0 ? (
            <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 space-y-1">
              {prefillWarnings.map((w) => <p key={w} className="text-[11px]" style={{ color: "#fbbf24" }}>{w}</p>)}
            </div>
          ) : null}

          {lastMessage ? (
            <div className="mt-3 rounded-lg border px-3 py-2 text-xs" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)", color: "var(--text-secondary)" }}>
              {lastMessage}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function Field({ label, value, onChange, disabled = false }: { label: string; value: string | number; onChange: (value: string) => void; disabled?: boolean }) {
  return (
    <label className="space-y-1">
      <span className="block text-[10px] font-bold uppercase tracking-[0.18em]" style={{ color: "var(--text-muted)" }}>{label}</span>
      <input
        value={String(value)}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border px-3 py-2 text-sm disabled:opacity-70"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }}
      />
    </label>
  );
}

function TextAreaField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="space-y-1">
      <span className="block text-[10px] font-bold uppercase tracking-[0.18em]" style={{ color: "var(--text-muted)" }}>{label}</span>
      <textarea
        rows={3}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border px-3 py-2 text-sm"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }}
      />
    </label>
  );
}
