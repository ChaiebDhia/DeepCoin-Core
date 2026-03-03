# frontend/Dockerfile
# ====================
# Multi-stage build for the DeepCoin Next.js frontend.
#
# WHY multi-stage:
#   Stage 1 (deps)     — install ALL node_modules (dev + prod)
#   Stage 2 (builder)  — run next build (outputs .next/standalone + .next/static)
#   Stage 3 (runner)   — lean Alpine image: only the standalone bundle (~40 MB)
#
# WHY node:20-alpine (not the full debian-based image):
#   The standalone Next.js bundle is a self-contained Node.js server.
#   It doesn't need npm, package.json, devDependencies, or the build cache.
#   Alpine gives a ~5× smaller final image (≈120 MB vs ≈600 MB).
#
# WHY output: "standalone" in next.config.ts:
#   Next.js traces every import at build time and produces .next/standalone —
#   a minimal copy of the app + only the node_modules it actually uses.
#   This is the "next build → deployable artifact" pattern recommended by Vercel.
#
# SECURITY:
#   - Non-root user (nextjs uid 1001) in the runner stage
#   - No .env.local baked in — secrets injected at runtime via compose env
#   - AUTH_SECRET and NEXTAUTH_SECRET go in via environment, not build args
#     (build args end up in the image layer cache — never put secrets there)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: deps — install node_modules with cache
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS deps

# libc6-compat is required for Next.js native bindings on Alpine
# (Alpine uses musl; some prebuilt binaries expect glibc compatibility shims)
RUN apk add --no-cache libc6-compat

WORKDIR /app

# Copy manifest files only — Docker caches this layer until they change.
# This means `npm ci` only reruns when package.json / package-lock.json change,
# not every time a source file changes.
COPY package.json package-lock.json* ./

# ci (clean install) is preferred in CI/container builds over `npm install`:
#   - Uses package-lock.json exactly (fully reproducible)
#   - Fails if the lock file is out of sync with package.json
#   - Faster than install because it skips lock file resolution
RUN npm ci

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: builder — compile the Next.js production bundle
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

RUN apk add --no-cache libc6-compat

WORKDIR /app

# Copy installed node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy all source files
COPY . .

# Build-time environment variables.
# NEXT_PUBLIC_* vars are inlined at build time — they must be set NOW.
# Auth secrets are runtime-only — never set them as build args.
ARG NEXT_PUBLIC_CLASSIFY_URL=http://nginx/api
ARG NEXT_PUBLIC_API_KEY=""
ENV NEXT_PUBLIC_CLASSIFY_URL=${NEXT_PUBLIC_CLASSIFY_URL} \
    NEXT_PUBLIC_API_KEY=${NEXT_PUBLIC_API_KEY} \
    NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production

# Run the Next.js production build.
# Requires output: "standalone" in next.config.ts to produce the slim bundle.
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: runner — minimal runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS runner

LABEL org.opencontainers.image.title="DeepCoin Web" \
      org.opencontainers.image.description="Next.js 15 frontend — standalone build" \
      org.opencontainers.image.authors="Dhia Chaieb <dhia.chaieb@esprit.tn>"

WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME="0.0.0.0"

# Non-root user
# WHY: if the container is compromised, an attacker cannot write to /etc, /app,
# or other host-mounted paths that are owned by root.
RUN addgroup --system --gid 1001 nodejs \
 && adduser  --system --uid 1001 nextjs

# Copy ONLY the artifacts needed to run the app:
#   .next/standalone — Next.js server + minimal node_modules
#   .next/static     — browser assets (JS, CSS)
#   public/          — static files (images, fonts, manifest)
#
# WHY copy public/static separately:
#   standalone/public and standalone/.next/static are symlinks in some builds.
#   Explicit COPY ensures the actual files are in the image layers.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static    ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public          ./public

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=15s \
    CMD wget -qO- http://localhost:3000/ || exit 1

# Start the standalone Next.js server.
# next/standalone builds output a self-contained server.js — no `next start` needed.
CMD ["node", "server.js"]
