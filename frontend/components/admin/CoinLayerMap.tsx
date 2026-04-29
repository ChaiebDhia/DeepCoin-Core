"use client";

import { useEffect, useMemo } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  Tooltip,
  useMap,
} from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";

import type { AdminCoinStatPoint } from "@/types/api";

type MapLayerFilter = "all" | "training" | "rag_only" | "user_added";

const LAYER_COLORS: Record<MapLayerFilter, string> = {
  all: "#94a3b8",
  training: "#3b82f6",
  rag_only: "#22c55e",
  user_added: "#f59e0b",
};

function labelForLayer(layer: string): string {
  if (layer === "training") return "Training";
  if (layer === "rag_only") return "RAG";
  if (layer === "user_added") return "User Added";
  return layer;
}

export function CoinLayerMap({
  points,
  layer,
}: {
  points: AdminCoinStatPoint[];
  layer: MapLayerFilter;
}) {
  const filtered = useMemo(() => {
    return points
      .filter((p) => p.latitude != null && p.longitude != null)
      .filter((p) => layer === "all" || p.layer === layer);
  }, [points, layer]);

  const withLayerOffset = (point: AdminCoinStatPoint): [number, number] => {
    const lat = Number(point.latitude);
    const lng = Number(point.longitude);
    if (point.layer === "training") return [lat + 0.06, lng - 0.06];
    if (point.layer === "rag_only") return [lat - 0.06, lng + 0.06];
    return [lat, lng];
  };

  const withJitter = (lat: number, lng: number, seed: string): [number, number] => {
    // Deterministic tiny jitter keeps overlapping markers individually clickable.
    let h = 0;
    for (let i = 0; i < seed.length; i += 1) h = (h * 31 + seed.charCodeAt(i)) | 0;
    const a = ((h % 360) * Math.PI) / 180;
    const r = 0.02 + (Math.abs(h) % 7) * 0.004;
    return [lat + Math.sin(a) * r, lng + Math.cos(a) * r];
  };

  const bounds = useMemo<LatLngBoundsExpression | null>(() => {
    if (!filtered.length) return null;
    return filtered.map((p) => {
      const base = withLayerOffset(p);
      return withJitter(base[0], base[1], `${p.layer}-${p.region}-${p.mint}-${p.count}`);
    });
  }, [filtered]);

  return (
    <div className="coin-layer-map relative isolate z-0 rounded-xl border p-3" style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-2)" }}>
      <div className="relative h-[420px] overflow-hidden rounded-lg border" style={{ borderColor: "var(--border)" }}>
        <MapContainer
          className="z-0"
          center={[39.0, 26.0]}
          zoom={4}
          minZoom={3}
          scrollWheelZoom
          style={{ height: "100%", width: "100%" }}
        >
          <MapAutoFit bounds={bounds} />
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {filtered.map((point, idx) => {
            const radiusBase = Math.max(3, Math.min(11, Math.sqrt(point.count + 1) * 0.85));
            const radius = point.layer === "user_added"
              ? Math.max(8, Math.min(16, radiusBase + 3))
              : radiusBase;
            const color = point.color || LAYER_COLORS[(point.layer as MapLayerFilter) || "all"] || "#94a3b8";
            const offset = withLayerOffset(point);
            const center = withJitter(offset[0], offset[1], `${point.layer}-${point.region}-${point.mint}-${idx}`);

            return (
              <CircleMarker
                key={`${point.layer}-${point.region}-${point.mint}-${point.latitude}-${point.longitude}-${idx}`}
                center={center}
                radius={radius}
                pathOptions={{ color, fillColor: color, fillOpacity: 0.58, weight: 1.4 }}
              >
                <Tooltip direction="top" offset={[0, -4]} opacity={0.95}>
                  <div style={{ fontSize: 11, fontWeight: 600 }}>
                    {point.mint || point.region || "Unknown location"} • {point.count}
                  </div>
                </Tooltip>
                <Popup>
                  <div style={{ minWidth: 180 }}>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>{point.region || point.mint || "Unknown location"}</div>
                    <div><strong>Layer:</strong> {labelForLayer(point.layer)}</div>
                    {point.mint ? <div><strong>Mint:</strong> {point.mint}</div> : null}
                    <div><strong>Coins:</strong> {point.count.toLocaleString()} in total</div>
                    <div><strong>Lat/Lon:</strong> {Number(point.latitude).toFixed(2)}, {Number(point.longitude).toFixed(2)}</div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>

      <div className="mt-3 flex flex-wrap gap-2 text-[11px]" style={{ color: "var(--text-muted)" }}>
        <Legend color={LAYER_COLORS.training} label="Training (CNN)" />
        <Legend color={LAYER_COLORS.rag_only} label="RAG-only corpus" />
        <Legend color={LAYER_COLORS.user_added} label="User-added inventory" />
      </div>
    </div>
  );
}

function MapAutoFit({ bounds }: { bounds: LatLngBoundsExpression | null }) {
  const map = useMap();
  useEffect(() => {
    if (!bounds || (bounds as [number, number][]).length === 0) return;
    map.fitBounds(bounds, { padding: [20, 20], maxZoom: 7 });
  }, [map, bounds]);
  return null;
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2 py-1" style={{ backgroundColor: `${color}22` }}>
      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}
