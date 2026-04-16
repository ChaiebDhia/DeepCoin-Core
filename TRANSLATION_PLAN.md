# Master Translation Plan (English -> French)

## Phase 1: Backend AI Chat ($DONE)
- **Objective:** Modify the `chat.py` endpoint and LLM prompt to respond in French when requested.
- **Tasks:**
  - Retrieve the `language` parameter from the frontend request payload.
  - Append a strict system instruction to the LLM context (e.g., "Répondez uniquement en français." or "You must reply entirely in French.").

## Phase 2: PDF Generation ($PENDING)
- **Objective:** Translate the generated analysis reports.
- **Tasks:**
  - Expand the translation dictionary inside `synthesis.py` to cover all headers, labels, and table keys.
  - Inject the language parameter down to the PDF generator.

## Phase 3: Frontend Infrastructure ($PENDING)
- **Objective:** Set up `next-intl` globally so React components can consume translations without crashing.
- **Tasks:**
  - Create `messages/en.json` and `messages/fr.json` dictionaries.
  - Update `next.config.ts` if needed (or i18n routing).
  - Wrap `app/layout.tsx` with `<NextIntlClientProvider>` to distribute the loaded messages.

## Phase 4: Automated Frontend Refactoring ($PENDING)
- **Objective:** Replace hardcoded strings across all `.tsx` files.
- **Tasks:**
  - Write a robust Python script to scan the `/frontend` directory.
  - Automatically extract English text, generate the corresponding `en.json` and `fr.json` keys.
  - Replace the text with `t('key')` and inject `const t = useTranslations(...)`.
  - Execute the script across sections (Hero, Dashboard, Navbar, etc.).