# cloudflare-vault

Cloudflare Worker for **Task #227 — Cloud Vault single-user test**.

Mirrors the browser IndexedDB Vault onto Cloudflare so we can find out
what breaks when the vault is remote: latency, R2 round-trips, Vectorize
accuracy, search-WASM compatibility. Single user, no orgs, no per-member
scoping. Layered org work lands in Task #198.

> ⚠️ The Vite/React app does **not** deploy this Worker. You deploy it
> manually with `wrangler`. The app talks to it over HTTPS via the
> `CloudflareWorkerAdapter` in `client/src/lib/vault/storage-adapter/`.

## Architecture

```
Browser  ──HTTPS──>  Worker (this package)  ─┬─> D1   (metadata)
                                             ├─> R2   (PDF bytes)
                                             ├─> Vectorize (embeddings)
                                             └─> Workers AI (embed model)
```

- **D1 (`vault-test-meta`)** — tables that mirror the IndexedDB stores
  1:1, plus a `chunk_vectors` table that records which Vectorize IDs
  belong to which doc so cascade-delete is deterministic.
- **R2 (`vault-test-bytes`)** — flat `pdfs/<docId>.pdf` layout for native
  PDF bytes only. The page-text cache stays browser-local (it's a
  render-time accelerator, not durable state).
- **Vectorize (`vault-test-vectors`)** — 384-dim cosine index. IDs are
  `<docId>:<chunkId>`. Metadata: `{ docId, chunkId, charStart, charEnd }`.
- **Workers AI** — `@cf/baai/bge-small-en-v1.5` for embeddings. Same model
  family the local transformers.js path uses so hit quality is comparable.

## One-shot provisioning

```bash
cd cloudflare-vault
npm install   # installs wrangler + types

wrangler login
wrangler r2 bucket create vault-test-bytes
wrangler d1 create vault-test-meta
#   ↑ copy the returned database_id into wrangler.toml
wrangler vectorize create vault-test-vectors --dimensions=384 --metric=cosine

# Shared bearer token (any high-entropy string; paste the same one into
# the browser settings panel's "Worker token" field):
openssl rand -hex 32 | wrangler secret put VAULT_API_TOKEN

# Apply the D1 schema (FTS5 + triggers + chunk_vectors):
npm run schema:remote
```

## Deploy

```bash
npm run deploy
```

Take the deployed URL (e.g. `https://vault-test-api.<your-subdomain>.workers.dev`)
and paste it into the **Vault settings → Cloudflare storage** panel in
the app, along with the bearer token.

## Local dev

```bash
npm run dev               # wrangler dev — local Worker on :8787
npm run schema:local      # apply schema to the local D1 file
```

## Endpoints (auth: `Authorization: Bearer <VAULT_API_TOKEN>` on every call)

| Method | Path | Notes |
| ------ | ---- | ----- |
| GET    | `/health` | Pings each binding. Public-but-rate-limited route — auth still required. |
| GET    | `/docs` | List documents. |
| GET    | `/docs/:id` | Get one. |
| PUT    | `/docs/:id` | Upsert. Body: `{ document: LocalSourceDocument }`. |
| DELETE | `/docs/:id` | Cascade: D1 rows + R2 object + Vectorize entries via `chunk_vectors`. |
| GET    | `/docs/:id/markdown` / PUT | Body: `{ markdown: LocalMarkdownDocument }`. |
| GET    | `/docs/:id/entries`  / PUT | Body: `{ entries: LocalContentEntry[] }`. Bulk replace. |
| GET    | `/docs/:id/content-map` / PUT | Body: `{ contentMap: LocalContentMapEntry[] }`. Bulk replace. |
| GET    | `/document-index` | List all `LocalDocumentIndex` rows. |
| PUT    | `/docs/:id/document-index` | Body: `{ documentIndex: LocalDocumentIndex }`. |
| GET    | `/docs/:id/pdf` | Streams the R2 object. Content-Type: `application/pdf`. |
| PUT    | `/docs/:id/pdf` | Single-PUT only, capped at 95 MB. Larger ⇒ 413. |
| DELETE | `/docs/:id/pdf` | Removes only the R2 object (cascade lives in DELETE /docs/:id). |
| POST   | `/search/keyword` | Body: `{ query, limit? }`. FTS5 if available, LIKE fallback. |
| POST   | `/search/semantic` | Body: `{ query, limit? }`. Vectorize + D1 hydration. |
| POST   | `/docs/:id/index` | (Re)chunk + (re)embed + Vectorize upsert. Returns `{ indexStatus, chunksIndexed, chunksFailed, errors[] }`. |

### CORS

`OPTIONS` preflight is handled on every endpoint. Allowed headers:
`Authorization, Content-Type, X-Vault-Checksum, X-Vault-Overwrite`. Methods:
`GET, PUT, POST, DELETE, OPTIONS`. Origin is governed by the
`ALLOWED_ORIGINS` env var in `wrangler.toml` (`*` by default for the
test deployment).

### Indexing is decoupled from metadata writes

`PUT /docs/:id/entries` returns immediately after the D1 write — it does
not embed inline. The client follows up with `POST /docs/:id/index` once
the entries (or content-map) are committed. This keeps document saves
fast even when Workers AI is rate-limited; per-chunk failures degrade
the `indexStatus` to `partial` instead of failing the whole request.

## Out of scope for this test

- Org / per-member scoping (Task #198 adds these additively).
- Multipart PDF uploads (single-PUT cap is the test-version constraint).
- Local-only Vault Sets (`localOnlyVaultSets`, set membership) — stay
  browser-local.
- Page-text cache + WASM viewer search — render-time accelerators that
  rebuild locally on viewer open.
