# DeepCoin Frontend (Next.js 15)

This frontend is the product UI for DeepCoin-Core, not a template app. It connects to the FastAPI backend for classification, report history, chat, auth flows, and admin tools.

## What this frontend includes

- Analyze flow with drag-drop uploader and mission-control agent pipeline
- 3-state CNN result UX (Identified / Consistent Match / Deep Search)
- Grad-CAM++ visualization card and report links
- History pages, detail pages, feedback capture, and admin dashboard
- Chat UI with streaming responses and source display
- i18n support (English/French) and light/dark themes

## Local development

From the `frontend/` directory:

```bash
npm install
npm run dev
```

Default local URL: `http://localhost:3000`

## Required environment variables

The frontend reads values from `.env.local` (or compose env in Docker).

```dotenv
# API endpoint used by browser in dev
NEXT_PUBLIC_CLASSIFY_URL=http://127.0.0.1:8000

# Enable/disable Google provider in the client bundle
NEXT_PUBLIC_GOOGLE_AUTH_ENABLED=1

# NextAuth + bridge wiring
NEXTAUTH_SECRET=
AUTH_BRIDGE_SECRET=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
```

## Docker notes

This project relies on build-time inlining for `NEXT_PUBLIC_*` variables.
If you change `NEXT_PUBLIC_GOOGLE_AUTH_ENABLED`, rebuild the frontend image.

```powershell
docker compose build web
docker compose up -d web
```

## Local-first AI mode (Ollama)

The frontend does not call Ollama directly. It calls the backend, and the backend decides provider routing.

To run local-first and avoid paid keys:

1. Start `ollama` service in compose.
2. Pull models in the Ollama container.
3. Keep `GITHUB_TOKEN` and `GOOGLE_API_KEY` unset in backend env.
4. Ensure `OLLAMA_HOST` is configured for backend.

## Build and type-check

```bash
npm run lint
npm run build
```

## Troubleshooting

- Google provider not visible:
	- Verify `NEXT_PUBLIC_GOOGLE_AUTH_ENABLED=1`
	- Rebuild frontend image after changing `NEXT_PUBLIC_*`
	- Confirm `/api/auth/providers` returns `google`
- Auth endpoint returns 502 behind nginx:
	- Restart nginx to refresh upstream resolution
- API calls fail on localhost:
	- Use `127.0.0.1` in `NEXT_PUBLIC_CLASSIFY_URL` to avoid IPv6 mismatch

## Recruiter quick checks

```powershell
# providers (google + credentials)
Invoke-WebRequest -UseBasicParsing http://localhost/api/auth/providers | Select-Object -ExpandProperty Content

# frontend health (if exposed by proxy)
Invoke-WebRequest -UseBasicParsing http://localhost | Select-Object -ExpandProperty StatusCode
```
