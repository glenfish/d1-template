/**
 * Task #227 — Cloudflare Worker for the single-user cloud vault test.
 *
 * One file on purpose: every endpoint, the CORS layer, bearer auth,
 * D1↔R2↔Vectorize↔Workers AI plumbing, and the chunk-and-embed pipeline
 * live together so reviewers can read the whole surface without jumping
 * between modules.
 *
 * Hard constraints (see task plan):
 *   - SINGLE USER. No `userId`/`orgId` columns. Task #198 layers those on.
 *   - INDEXING IS DECOUPLED FROM METADATA WRITES. `PUT /docs/:id/entries`
 *     never embeds inline; the client calls `POST /docs/:id/index` after.
 *   - DELETE /docs/:id MUST enumerate `chunk_vectors` and pass exact IDs to
 *     `VECTORS.deleteByIds()`. No reliance on metadata-filter scans.
 *   - PDF uploads are single-PUT only, capped at `PDF_SINGLE_PUT_MAX_BYTES`.
 *   - CORS preflight is handled on every route — verified end-to-end at
 *     `/health` before any other endpoint was added.
 *   - FTS5 is used for `/search/keyword` when the schema applied cleanly;
 *     a LIKE fallback returns the same response shape if it didn't.
 */

export interface Env {
  AI: Ai;
  PDF_BUCKET: R2Bucket;
  META_DB: D1Database;
  VECTORS: VectorizeIndex;
  VAULT_API_TOKEN: string;
  ALLOWED_ORIGINS: string;
  PDF_SINGLE_PUT_MAX_BYTES: string;
  EMBEDDING_MODEL: string;
  EMBEDDING_DIMENSIONS: string;
}

// ---------- HTTP helpers --------------------------------------------------

const ALLOWED_METHODS = "GET, PUT, POST, DELETE, OPTIONS";
const ALLOWED_HEADERS = "Authorization, Content-Type, X-Vault-Checksum, X-Vault-Overwrite";

// ---------- Typed shapes for inbound JSON bodies + AI client -------------
//
// Inbound bodies are user-controlled JSON, so we describe them as
// `Partial<…>` and validate field-by-field rather than reaching for `any`.
// The Workers AI client is exposed via a narrow run(model, input) shape
// that matches the bge-small-en-v1.5 contract we actually use; no `any`
// cast, no JSDoc lies.

interface InboundDocumentBody {
  id?: string;
  sourceType?: string;
  sourceImportId?: string;
  sourceDocumentId?: string;
  title?: string;
  createdAt?: string;
  updatedAt?: string;
  importedAt?: string;
  entryCount?: number;
  checksum?: string;
  sizeBytes?: number;
  parserVersion?: string;
  indexVersion?: string;
  reindexRequired?: boolean;
  deletedAt?: string | null;
}

interface InboundEntryBody {
  id?: string;
  documentId?: string;
  role?: string;
  content?: string;
  heading?: string;
  headingLevel?: number;
  orderIndex?: number;
  markdownStartOffset?: number;
  markdownEndOffset?: number;
  offsetEncoding?: string;
  createdAt?: string;
  metadata?: unknown;
}

interface InboundContentMapBody {
  id?: string;
  documentId?: string;
  segmentNumber?: number;
  entryIds?: unknown;
  headingPreview?: string;
  contentHead?: string;
  contentTail?: string;
  responseHead?: string;
  responseTail?: string;
  keywords?: unknown;
  entryStartIndex?: number;
  entryEndIndex?: number;
  indexText?: string;
  anchor?: unknown;
  indexVersion?: string;
  createdAt?: string;
}

interface InboundDocIndexBody {
  documentId?: string;
  sourceType?: string;
  title?: string;
  createdAt?: string;
  updatedAt?: string;
  entryCount?: number;
  segmentCount?: number;
  keywordFingerprint?: unknown;
  indexTextPreview?: string;
  indexVersion?: string;
  generatedAt?: string;
}

interface InboundImportJobBody {
  id?: string;
  fileName?: string;
  fileSize?: number;
  detectedSourceType?: string;
  status?: string;
  progress?: number;
  currentStepLabel?: string;
  documentsProcessed?: number;
  entriesProcessed?: number;
  documentIds?: string[];
  error?: string;
  createdAt?: string;
  updatedAt?: string;
}

interface ImportJobRow {
  id: string;
  file_name: string;
  file_size: number;
  detected_source_type: string | null;
  status: string;
  progress: number;
  current_step_label: string | null;
  documents_processed: number;
  entries_processed: number;
  document_ids: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

function rowToImportJob(r: ImportJobRow): Record<string, unknown> {
  return {
    id: r.id,
    fileName: r.file_name,
    fileSize: r.file_size,
    detectedSourceType: r.detected_source_type ?? undefined,
    status: r.status,
    progress: r.progress,
    currentStepLabel: r.current_step_label ?? undefined,
    documentsProcessed: r.documents_processed,
    entriesProcessed: r.entries_processed,
    documentIds: r.document_ids ? safeParseJSON(r.document_ids) ?? [] : undefined,
    error: r.error ?? undefined,
    createdAt: r.created_at,
    updatedAt: r.updated_at,
  };
}

interface SegmentSearchRow {
  id: string;
  document_id: string;
  segment_number: number;
  heading_preview: string | null;
  content_head: string | null;
  title: string | null;
  score?: number;
}

interface EntrySearchRow {
  id: string;
  document_id: string;
  heading: string | null;
  content: string | null;
  title: string | null;
  score?: number;
}

interface VectorChunkMetadata {
  docId: string;
  chunkId: string;
  snippet: string;
  charStart: number;
  charEnd: number;
}

interface WorkersAiResponse { data: number[][] }
interface WorkersAiClient {
  run(model: string, input: { text: string[] }): Promise<WorkersAiResponse>;
}

function corsHeaders(env: Env, request: Request): Record<string, string> {
  const origin = request.headers.get("Origin") ?? "";
  const allowList = (env.ALLOWED_ORIGINS || "*")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const allowOrigin =
    allowList.includes("*") || allowList.includes(origin) ? (allowList.includes("*") ? "*" : origin) : "";
  const headers: Record<string, string> = {
    "Access-Control-Allow-Methods": ALLOWED_METHODS,
    "Access-Control-Allow-Headers": ALLOWED_HEADERS,
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
  if (allowOrigin) headers["Access-Control-Allow-Origin"] = allowOrigin;
  return headers;
}

function jsonResponse(
  body: unknown,
  init: ResponseInit = {},
  env?: Env,
  request?: Request,
): Response {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (env && request) {
    for (const [k, v] of Object.entries(corsHeaders(env, request))) {
      headers.set(k, v);
    }
  }
  return new Response(JSON.stringify(body), { ...init, headers });
}

function errorResponse(
  status: number,
  message: string,
  env: Env,
  request: Request,
): Response {
  return jsonResponse({ error: message }, { status }, env, request);
}

function checkAuth(request: Request, env: Env): string | null {
  const auth = request.headers.get("Authorization") ?? "";
  if (!auth.startsWith("Bearer ")) return "Missing bearer token.";
  const token = auth.slice("Bearer ".length).trim();
  if (!env.VAULT_API_TOKEN) return "Server token not configured.";
  if (token !== env.VAULT_API_TOKEN) return "Bearer token rejected.";
  return null;
}

// ---------- Row marshalling (D1 ↔ Local* types) ---------------------------

interface DocRow {
  id: string;
  source_type: string;
  source_import_id: string;
  source_document_id: string | null;
  title: string | null;
  created_at: string | null;
  updated_at: string | null;
  imported_at: string;
  entry_count: number;
  checksum: string;
  size_bytes: number | null;
  parser_version: string;
  index_version: string | null;
  reindex_required: number;
  deleted_at: string | null;
}

function rowToDocument(r: DocRow): Record<string, unknown> {
  return {
    id: r.id,
    sourceType: r.source_type,
    sourceImportId: r.source_import_id,
    sourceDocumentId: r.source_document_id ?? undefined,
    title: r.title ?? undefined,
    createdAt: r.created_at ?? undefined,
    updatedAt: r.updated_at ?? undefined,
    importedAt: r.imported_at,
    entryCount: r.entry_count,
    checksum: r.checksum,
    sizeBytes: r.size_bytes ?? undefined,
    parserVersion: r.parser_version,
    indexVersion: r.index_version ?? undefined,
    reindexRequired: r.reindex_required === 1,
    deletedAt: r.deleted_at ?? undefined,
  };
}

function documentToRow(d: InboundDocumentBody): DocRow {
  return {
    id: String(d.id),
    source_type: String(d.sourceType),
    source_import_id: String(d.sourceImportId),
    source_document_id: d.sourceDocumentId ?? null,
    title: d.title ?? null,
    created_at: d.createdAt ?? null,
    updated_at: d.updatedAt ?? null,
    imported_at: String(d.importedAt),
    entry_count: Number(d.entryCount ?? 0),
    checksum: String(d.checksum),
    size_bytes: d.sizeBytes ?? null,
    parser_version: String(d.parserVersion),
    index_version: d.indexVersion ?? null,
    reindex_required: d.reindexRequired ? 1 : 0,
    deleted_at: d.deletedAt ?? null,
  };
}

interface EntryRow {
  id: string;
  document_id: string;
  role: string | null;
  content: string;
  heading: string | null;
  heading_level: number | null;
  order_index: number;
  markdown_start_offset: number;
  markdown_end_offset: number;
  offset_encoding: string;
  created_at: string | null;
  metadata: string | null;
}

function rowToEntry(r: EntryRow): Record<string, unknown> {
  return {
    id: r.id,
    documentId: r.document_id,
    role: r.role ?? undefined,
    content: r.content,
    heading: r.heading ?? undefined,
    headingLevel: r.heading_level ?? undefined,
    orderIndex: r.order_index,
    markdownStartOffset: r.markdown_start_offset,
    markdownEndOffset: r.markdown_end_offset,
    offsetEncoding: r.offset_encoding,
    createdAt: r.created_at ?? undefined,
    metadata: r.metadata ? safeParseJSON(r.metadata) : undefined,
  };
}

function safeParseJSON(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return undefined;
  }
}

interface ContentMapRow {
  id: string;
  document_id: string;
  segment_number: number;
  entry_ids: string;
  heading_preview: string | null;
  content_head: string;
  content_tail: string | null;
  response_head: string | null;
  response_tail: string | null;
  keywords: string;
  entry_start_index: number;
  entry_end_index: number;
  index_text: string;
  anchor: string;
  index_version: string;
  created_at: string;
}

function rowToContentMap(r: ContentMapRow): Record<string, unknown> {
  return {
    id: r.id,
    documentId: r.document_id,
    segmentNumber: r.segment_number,
    entryIds: safeParseJSON(r.entry_ids) ?? [],
    headingPreview: r.heading_preview ?? undefined,
    contentHead: r.content_head,
    contentTail: r.content_tail ?? undefined,
    responseHead: r.response_head ?? undefined,
    responseTail: r.response_tail ?? undefined,
    keywords: safeParseJSON(r.keywords) ?? [],
    entryStartIndex: r.entry_start_index,
    entryEndIndex: r.entry_end_index,
    indexText: r.index_text,
    anchor: safeParseJSON(r.anchor) ?? {},
    indexVersion: r.index_version,
    createdAt: r.created_at,
  };
}

interface DocIndexRow {
  document_id: string;
  source_type: string;
  title: string | null;
  created_at: string | null;
  updated_at: string | null;
  entry_count: number;
  segment_count: number;
  keyword_fingerprint: string;
  index_text_preview: string;
  index_version: string;
  generated_at: string;
}

function rowToDocIndex(r: DocIndexRow): Record<string, unknown> {
  return {
    documentId: r.document_id,
    sourceType: r.source_type,
    title: r.title ?? undefined,
    createdAt: r.created_at ?? undefined,
    updatedAt: r.updated_at ?? undefined,
    entryCount: r.entry_count,
    segmentCount: r.segment_count,
    keywordFingerprint: safeParseJSON(r.keyword_fingerprint) ?? [],
    indexTextPreview: r.index_text_preview,
    indexVersion: r.index_version,
    generatedAt: r.generated_at,
  };
}

// ---------- Route dispatcher ---------------------------------------------

interface RouteMatch {
  method: string;
  pattern: RegExp;
  handler: (
    request: Request,
    env: Env,
    ctx: ExecutionContext,
    match: RegExpMatchArray,
  ) => Promise<Response>;
}

function buildRoutes(): RouteMatch[] {
  return [
    { method: "GET", pattern: /^\/health$/, handler: handleHealth },
    { method: "GET", pattern: /^\/docs$/, handler: handleListDocs },
    { method: "GET", pattern: /^\/docs\/([^/]+)$/, handler: handleGetDoc },
    { method: "PUT", pattern: /^\/docs\/([^/]+)$/, handler: handlePutDoc },
    { method: "DELETE", pattern: /^\/docs\/([^/]+)$/, handler: handleDeleteDoc },
    { method: "GET", pattern: /^\/docs\/([^/]+)\/markdown$/, handler: handleGetMarkdown },
    { method: "PUT", pattern: /^\/docs\/([^/]+)\/markdown$/, handler: handlePutMarkdown },
    { method: "GET", pattern: /^\/docs\/([^/]+)\/entries$/, handler: handleGetEntries },
    { method: "PUT", pattern: /^\/docs\/([^/]+)\/entries$/, handler: handlePutEntries },
    { method: "GET", pattern: /^\/docs\/([^/]+)\/content-map$/, handler: handleGetContentMap },
    { method: "PUT", pattern: /^\/docs\/([^/]+)\/content-map$/, handler: handlePutContentMap },
    { method: "GET", pattern: /^\/document-index$/, handler: handleListDocIndex },
    { method: "PUT", pattern: /^\/docs\/([^/]+)\/document-index$/, handler: handlePutDocIndex },
    { method: "GET", pattern: /^\/docs\/([^/]+)\/pdf$/, handler: handleGetPdf },
    { method: "PUT", pattern: /^\/docs\/([^/]+)\/pdf$/, handler: handlePutPdf },
    { method: "DELETE", pattern: /^\/docs\/([^/]+)\/pdf$/, handler: handleDeletePdf },
    { method: "POST", pattern: /^\/search\/keyword$/, handler: handleSearchKeyword },
    { method: "POST", pattern: /^\/search\/semantic$/, handler: handleSearchSemantic },
    { method: "POST", pattern: /^\/docs\/([^/]+)\/index$/, handler: handleIndexDoc },
    { method: "GET", pattern: /^\/jobs\/status$/, handler: handleJobsStatus },
    { method: "DELETE", pattern: /^\/jobs$/, handler: handleClearJobs },
    { method: "GET", pattern: /^\/coverage$/, handler: handleCoverage },
    { method: "GET", pattern: /^\/import-jobs$/, handler: handleListImportJobs },
    { method: "GET", pattern: /^\/import-jobs\/([^/]+)$/, handler: handleGetImportJob },
    { method: "PUT", pattern: /^\/import-jobs\/([^/]+)$/, handler: handlePutImportJob },
    { method: "DELETE", pattern: /^\/import-jobs\/([^/]+)$/, handler: handleDeleteImportJob },
  ];
}

const ROUTES = buildRoutes();

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(env, request) });
    }
    const authError = checkAuth(request, env);
    if (authError) return errorResponse(401, authError, env, request);

    const url = new URL(request.url);
    const path = url.pathname;
    for (const route of ROUTES) {
      if (route.method !== request.method) continue;
      const m = path.match(route.pattern);
      if (m) {
        try {
          return await route.handler(request, env, ctx, m);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          console.error(`[vault-worker] ${route.method} ${path} failed:`, msg);
          return errorResponse(500, msg, env, request);
        }
      }
    }
    return errorResponse(404, `No route for ${request.method} ${path}`, env, request);
  },
};

// ---------- Handlers ------------------------------------------------------

async function handleHealth(request: Request, env: Env): Promise<Response> {
  const bindings = { d1: false, r2: false, vectorize: false, ai: false };
  try {
    const r = await env.META_DB.prepare("SELECT 1 AS ok").first<{ ok: number }>();
    bindings.d1 = r?.ok === 1;
  } catch (e) {
    console.warn("[health] D1 ping failed:", e);
  }
  try {
    if (!env.PDF_BUCKET || typeof env.PDF_BUCKET.head !== "function") {
      throw new Error("PDF_BUCKET binding missing or invalid");
    }
    // head() returns null for a missing key without throwing; reaching this
    // line means the binding is wired correctly.
    await env.PDF_BUCKET.head("__health__");
    bindings.r2 = true;
  } catch (e) {
    console.warn("[health] R2 probe failed:", e);
    bindings.r2 = false;
  }
  try {
    await env.VECTORS.describe();
    bindings.vectorize = true;
  } catch (e) {
    console.warn("[health] Vectorize describe failed:", e);
  }
  bindings.ai = !!env.AI;
  const ok = bindings.d1 && bindings.r2 && bindings.vectorize && bindings.ai;
  return jsonResponse({ ok, bindings }, {}, env, request);
}

async function handleListDocs(request: Request, env: Env): Promise<Response> {
  const r = await env.META_DB.prepare(
    "SELECT * FROM documents WHERE deleted_at IS NULL ORDER BY imported_at DESC",
  ).all<DocRow>();
  const documents = (r.results ?? []).map(rowToDocument);
  return jsonResponse({ documents }, {}, env, request);
}

async function handleGetDoc(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const row = await env.META_DB.prepare("SELECT * FROM documents WHERE id = ?").bind(id).first<DocRow>();
  if (!row) return errorResponse(404, `Document ${id} not found.`, env, request);
  return jsonResponse({ document: rowToDocument(row) }, {}, env, request);
}

async function handlePutDoc(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const body = (await request.json()) as { document?: InboundDocumentBody };
  const doc = body.document;
  if (!doc || doc.id !== id) {
    return errorResponse(400, "Request body must include `document` with matching id.", env, request);
  }
  // Mirror integrity contract — two layers:
  //   (1) X-Vault-Checksum must equal body.checksum (catches garbled
  //       bodies / header-body mismatch / proxy tampering).
  //   (2) If a row already exists with a DIFFERENT checksum, the client
  //       must explicitly opt into the overwrite by sending
  //       `X-Vault-Overwrite: 1` (or the prior server-side checksum it
  //       expected to find). Otherwise we 409, so a stale tab can't
  //       silently clobber a freshly-mirrored copy. Idempotent re-writes
  //       of identical content (same checksum) always succeed.
  const headerChecksum = request.headers.get("X-Vault-Checksum");
  if (headerChecksum && doc.checksum && headerChecksum !== doc.checksum) {
    return errorResponse(
      400,
      `X-Vault-Checksum (${headerChecksum}) does not match document.checksum (${doc.checksum}).`,
      env, request,
    );
  }
  const existing = await env.META_DB
    .prepare("SELECT checksum FROM documents WHERE id = ?")
    .bind(id)
    .first<{ checksum: string }>();
  if (existing && doc.checksum && existing.checksum !== doc.checksum) {
    const overwriteHeader = request.headers.get("X-Vault-Overwrite") ?? "";
    const allowOverwrite =
      overwriteHeader === "1" || overwriteHeader === existing.checksum;
    if (!allowOverwrite) {
      return errorResponse(
        409,
        `Document ${id} exists with checksum ${existing.checksum}; refusing to overwrite with ${doc.checksum}. Set X-Vault-Overwrite: 1 or the prior checksum to acknowledge.`,
        env, request,
      );
    }
  }
  const r = documentToRow(doc);
  await env.META_DB.prepare(
    `INSERT INTO documents (id, source_type, source_import_id, source_document_id, title,
        created_at, updated_at, imported_at, entry_count, checksum, size_bytes,
        parser_version, index_version, reindex_required, deleted_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
        source_type=excluded.source_type, source_import_id=excluded.source_import_id,
        source_document_id=excluded.source_document_id, title=excluded.title,
        created_at=excluded.created_at, updated_at=excluded.updated_at,
        imported_at=excluded.imported_at, entry_count=excluded.entry_count,
        checksum=excluded.checksum, size_bytes=excluded.size_bytes,
        parser_version=excluded.parser_version, index_version=excluded.index_version,
        reindex_required=excluded.reindex_required, deleted_at=excluded.deleted_at`,
  )
    .bind(
      r.id, r.source_type, r.source_import_id, r.source_document_id, r.title,
      r.created_at, r.updated_at, r.imported_at, r.entry_count, r.checksum, r.size_bytes,
      r.parser_version, r.index_version, r.reindex_required, r.deleted_at,
    )
    .run();
  return jsonResponse({ ok: true }, {}, env, request);
}

async function handleDeleteDoc(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  // Step 1: enumerate exact Vectorize IDs from chunk_vectors. This is the
  // deterministic source of truth — we never rely on metadata-filter scans.
  const vectorRows = await env.META_DB
    .prepare("SELECT vector_id FROM chunk_vectors WHERE document_id = ?")
    .bind(id)
    .all<{ vector_id: string }>();
  const vectorIds = (vectorRows.results ?? []).map((r) => r.vector_id);
  if (vectorIds.length > 0) {
    // Transactionally blocking: if Vectorize delete fails, do NOT wipe
    // `chunk_vectors` — that's our deterministic source-of-truth for
    // retrying the cleanup. Wiping it on failure would permanently
    // orphan the live vectors and leak ghost semantic hits.
    try {
      await env.VECTORS.deleteByIds(vectorIds);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.warn(`[delete] Vectorize.deleteByIds failed for ${id}:`, e);
      return errorResponse(
        502,
        `Vectorize delete failed for ${id} (${msg}). chunk_vectors retained for retry.`,
        env, request,
      );
    }
  }
  // Step 2: drop R2 PDF object (idempotent).
  try {
    await env.PDF_BUCKET.delete(`pdfs/${id}.pdf`);
  } catch (e) {
    console.warn(`[delete] R2 delete failed for ${id}:`, e);
  }
  // Step 3: cascade across all D1 tables in one batch.
  await env.META_DB.batch([
    env.META_DB.prepare("DELETE FROM chunk_vectors       WHERE document_id = ?").bind(id),
    env.META_DB.prepare("DELETE FROM chunk_embedding_jobs WHERE document_id = ?").bind(id),
    env.META_DB.prepare("DELETE FROM document_index      WHERE document_id = ?").bind(id),
    env.META_DB.prepare("DELETE FROM content_map          WHERE document_id = ?").bind(id),
    env.META_DB.prepare("DELETE FROM markdown             WHERE document_id = ?").bind(id),
    env.META_DB.prepare("DELETE FROM entries              WHERE document_id = ?").bind(id),
    env.META_DB.prepare("DELETE FROM documents            WHERE id = ?").bind(id),
  ]);
  return jsonResponse({ ok: true, vectorsDeleted: vectorIds.length }, {}, env, request);
}

async function handleGetMarkdown(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const row = await env.META_DB
    .prepare("SELECT * FROM markdown WHERE document_id = ?")
    .bind(id)
    .first<{ document_id: string; markdown: string; generated_at: string }>();
  if (!row) return errorResponse(404, `Markdown for ${id} not found.`, env, request);
  return jsonResponse(
    { markdown: { documentId: row.document_id, markdown: row.markdown, generatedAt: row.generated_at } },
    {}, env, request,
  );
}

async function handlePutMarkdown(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const body = (await request.json()) as { markdown?: { documentId: string; markdown: string; generatedAt: string } };
  const md = body.markdown;
  if (!md || md.documentId !== id) {
    return errorResponse(400, "Body must include `markdown` with matching documentId.", env, request);
  }
  await env.META_DB
    .prepare(
      `INSERT INTO markdown (document_id, markdown, generated_at) VALUES (?, ?, ?)
       ON CONFLICT(document_id) DO UPDATE SET markdown=excluded.markdown, generated_at=excluded.generated_at`,
    )
    .bind(md.documentId, md.markdown, md.generatedAt)
    .run();
  return jsonResponse({ ok: true }, {}, env, request);
}

async function handleGetEntries(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const r = await env.META_DB
    .prepare("SELECT * FROM entries WHERE document_id = ? ORDER BY order_index ASC")
    .bind(id)
    .all<EntryRow>();
  const entries = (r.results ?? []).map(rowToEntry);
  return jsonResponse({ entries }, {}, env, request);
}

async function handlePutEntries(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const body = (await request.json()) as { entries?: InboundEntryBody[] };
  const entries = body.entries ?? [];
  // Bulk-replace semantics matches the IDB pattern: caller computed the full
  // entry set for the doc, this endpoint just stores it atomically.
  const stmts: D1PreparedStatement[] = [
    env.META_DB.prepare("DELETE FROM entries WHERE document_id = ?").bind(id),
  ];
  for (const e of entries) {
    if (e.documentId !== id) {
      return errorResponse(400, `Entry ${e.id} documentId does not match URL.`, env, request);
    }
    stmts.push(
      env.META_DB
        .prepare(
          `INSERT INTO entries
            (id, document_id, role, content, heading, heading_level, order_index,
             markdown_start_offset, markdown_end_offset, offset_encoding, created_at, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .bind(
          String(e.id), id, e.role ?? null, String(e.content),
          e.heading ?? null, e.headingLevel ?? null, Number(e.orderIndex),
          Number(e.markdownStartOffset), Number(e.markdownEndOffset),
          String(e.offsetEncoding ?? "utf16"), e.createdAt ?? null,
          e.metadata ? JSON.stringify(e.metadata) : null,
        ),
    );
  }
  await env.META_DB.batch(stmts);
  return jsonResponse({ ok: true, entryCount: entries.length }, {}, env, request);
}

async function handleGetContentMap(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const r = await env.META_DB
    .prepare("SELECT * FROM content_map WHERE document_id = ? ORDER BY segment_number ASC")
    .bind(id)
    .all<ContentMapRow>();
  return jsonResponse({ contentMap: (r.results ?? []).map(rowToContentMap) }, {}, env, request);
}

async function handlePutContentMap(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const body = (await request.json()) as { contentMap?: InboundContentMapBody[] };
  const segments = body.contentMap ?? [];
  const stmts: D1PreparedStatement[] = [
    env.META_DB.prepare("DELETE FROM content_map WHERE document_id = ?").bind(id),
  ];
  for (const s of segments) {
    if (s.documentId !== id) {
      return errorResponse(400, `Segment ${s.id} documentId does not match URL.`, env, request);
    }
    stmts.push(
      env.META_DB
        .prepare(
          `INSERT INTO content_map
            (id, document_id, segment_number, entry_ids, heading_preview, content_head,
             content_tail, response_head, response_tail, keywords, entry_start_index,
             entry_end_index, index_text, anchor, index_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        )
        .bind(
          String(s.id), id, Number(s.segmentNumber),
          JSON.stringify(s.entryIds ?? []),
          s.headingPreview ?? null, String(s.contentHead),
          s.contentTail ?? null, s.responseHead ?? null, s.responseTail ?? null,
          JSON.stringify(s.keywords ?? []),
          Number(s.entryStartIndex), Number(s.entryEndIndex),
          String(s.indexText), JSON.stringify(s.anchor ?? {}),
          String(s.indexVersion), String(s.createdAt),
        ),
    );
  }
  await env.META_DB.batch(stmts);
  return jsonResponse({ ok: true, segmentCount: segments.length }, {}, env, request);
}

async function handleListDocIndex(request: Request, env: Env): Promise<Response> {
  const r = await env.META_DB.prepare("SELECT * FROM document_index").all<DocIndexRow>();
  return jsonResponse({ documentIndex: (r.results ?? []).map(rowToDocIndex) }, {}, env, request);
}

async function handlePutDocIndex(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const body = (await request.json()) as { documentIndex?: InboundDocIndexBody };
  const di = body.documentIndex;
  if (!di || di.documentId !== id) {
    return errorResponse(400, "Body must include `documentIndex` with matching documentId.", env, request);
  }
  await env.META_DB
    .prepare(
      `INSERT INTO document_index
        (document_id, source_type, title, created_at, updated_at, entry_count,
         segment_count, keyword_fingerprint, index_text_preview, index_version, generated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(document_id) DO UPDATE SET
         source_type=excluded.source_type, title=excluded.title,
         created_at=excluded.created_at, updated_at=excluded.updated_at,
         entry_count=excluded.entry_count, segment_count=excluded.segment_count,
         keyword_fingerprint=excluded.keyword_fingerprint,
         index_text_preview=excluded.index_text_preview,
         index_version=excluded.index_version, generated_at=excluded.generated_at`,
    )
    .bind(
      String(di.documentId), String(di.sourceType), di.title ?? null,
      di.createdAt ?? null, di.updatedAt ?? null, Number(di.entryCount ?? 0),
      Number(di.segmentCount ?? 0), JSON.stringify(di.keywordFingerprint ?? []),
      String(di.indexTextPreview ?? ""), String(di.indexVersion ?? ""), String(di.generatedAt),
    )
    .run();
  return jsonResponse({ ok: true }, {}, env, request);
}

async function handleGetPdf(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const obj = await env.PDF_BUCKET.get(`pdfs/${id}.pdf`);
  if (!obj) return errorResponse(404, `PDF for ${id} not found.`, env, request);
  const headers = new Headers();
  for (const [k, v] of Object.entries(corsHeaders(env, request))) headers.set(k, v);
  headers.set("Content-Type", "application/pdf");
  if (obj.size) headers.set("Content-Length", String(obj.size));
  if (obj.etag) headers.set("ETag", obj.etag);
  return new Response(obj.body, { status: 200, headers });
}

async function handlePutPdf(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const cap = Number(env.PDF_SINGLE_PUT_MAX_BYTES || 99614720);
  const lenHeader = Number(request.headers.get("Content-Length") || 0);
  if (lenHeader > cap) {
    return errorResponse(413, `PDF exceeds the ${cap}-byte single-PUT cap.`, env, request);
  }
  // PDF integrity contract: the client must send X-Vault-Checksum =
  // the source document's content checksum (the same value stored in
  // documents.checksum). If the documents row exists, we require the
  // header to match; this prevents a mirror runner from uploading a
  // PDF whose corresponding metadata row drifted (e.g. an aborted
  // earlier mirror left the row at version A while the client now
  // tries to upload the PDF for version B).
  const headerChecksum = request.headers.get("X-Vault-Checksum");
  const docRow = await env.META_DB
    .prepare("SELECT checksum FROM documents WHERE id = ?")
    .bind(id)
    .first<{ checksum: string }>();
  if (docRow) {
    if (!headerChecksum) {
      return errorResponse(
        400,
        `PDF PUT for existing document ${id} requires X-Vault-Checksum header (strict CAS).`,
        env, request,
      );
    }
    if (headerChecksum !== docRow.checksum) {
      return errorResponse(
        409,
        `PDF X-Vault-Checksum (${headerChecksum}) does not match documents.checksum (${docRow.checksum}). Upload the document metadata first or pass the matching checksum.`,
        env, request,
      );
    }
  }
  const buf = await request.arrayBuffer();
  if (buf.byteLength > cap) {
    return errorResponse(413, `PDF exceeds the ${cap}-byte single-PUT cap.`, env, request);
  }
  await env.PDF_BUCKET.put(`pdfs/${id}.pdf`, buf, {
    httpMetadata: { contentType: "application/pdf" },
  });
  return jsonResponse({ ok: true, bytes: buf.byteLength }, {}, env, request);
}

async function handleDeletePdf(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  await env.PDF_BUCKET.delete(`pdfs/${id}.pdf`);
  return jsonResponse({ ok: true }, {}, env, request);
}

// ---------- Search --------------------------------------------------------

async function handleSearchKeyword(request: Request, env: Env): Promise<Response> {
  const body = (await request.json()) as { query?: string; limit?: number };
  const query = (body.query ?? "").trim();
  const limit = Math.min(100, Math.max(1, body.limit ?? 25));
  if (!query) return jsonResponse({ hits: [] }, {}, env, request);

  // Try FTS5 first; fall back to LIKE if the virtual table is absent
  // (D1 instances created before FTS5 was enabled in the chosen
  // compatibility date won't have it).
  let hits: Array<Record<string, unknown>> = [];
  try {
    const ftsQuery = query
      .replace(/[^\p{L}\p{N}\s]/gu, " ")
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .map((t) => `${t}*`)
      .join(" ");
    if (ftsQuery) {
      const segR = await env.META_DB
        .prepare(
          `SELECT cm.id AS id, cm.document_id AS document_id, cm.segment_number AS segment_number,
                  cm.heading_preview AS heading_preview, cm.content_head AS content_head,
                  d.title AS title, bm25(content_map_fts) AS score
             FROM content_map_fts
             JOIN content_map cm ON cm.rowid = content_map_fts.rowid
             JOIN documents d ON d.id = cm.document_id
            WHERE content_map_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?`,
        )
        .bind(ftsQuery, limit)
        .all<SegmentSearchRow>();
      for (const r of segR.results ?? []) {
        hits.push({
          kind: "segment",
          documentId: r.document_id,
          segmentId: r.id,
          segmentNumber: r.segment_number,
          title: r.title ?? "Untitled",
          snippet: r.heading_preview || (r.content_head ?? "").slice(0, 280),
          score: 1 / (1 + Number(r.score ?? 1)),
        });
      }
      // Union with entries_fts so terms that live in the indexed entries
      // (importer output) but not yet in content_map are also matched.
      // The wrapping union here keeps response shape stable (kind:"entry").
      try {
        const entryR = await env.META_DB
          .prepare(
            `SELECT e.id AS id, e.document_id AS document_id, e.heading AS heading,
                    e.content AS content, d.title AS title, bm25(entries_fts) AS score
               FROM entries_fts
               JOIN entries e ON e.rowid = entries_fts.rowid
               JOIN documents d ON d.id = e.document_id
              WHERE entries_fts MATCH ?
              ORDER BY score ASC
              LIMIT ?`,
          )
          .bind(ftsQuery, limit)
          .all<EntrySearchRow>();
        for (const r of entryR.results ?? []) {
          hits.push({
            kind: "entry",
            documentId: r.document_id,
            entryId: r.id,
            title: r.title ?? "Untitled",
            snippet: (r.heading || r.content || "").slice(0, 280),
            score: 1 / (1 + Number(r.score ?? 1)),
          });
        }
      } catch (e) {
        console.warn("[search/keyword] entries_fts path failed:", e);
      }
    }
  } catch (e) {
    console.warn("[search/keyword] FTS5 path failed, falling back to LIKE:", e);
    hits = [];
  }

  if (hits.length === 0) {
    const like = `%${query.replace(/[%_]/g, " ")}%`;
    const r = await env.META_DB
      .prepare(
        `SELECT cm.id, cm.document_id, cm.segment_number, cm.heading_preview, cm.content_head,
                d.title
           FROM content_map cm
           JOIN documents d ON d.id = cm.document_id
          WHERE cm.index_text LIKE ? OR cm.content_head LIKE ?
          LIMIT ?`,
      )
      .bind(like, like, limit)
      .all<SegmentSearchRow>();
    for (const row of r.results ?? []) {
      hits.push({
        kind: "segment",
        documentId: row.document_id,
        segmentId: row.id,
        segmentNumber: row.segment_number,
        title: row.title ?? "Untitled",
        snippet: row.heading_preview || (row.content_head ?? "").slice(0, 280),
        score: 0.5,
      });
    }
    // Entries parity for the LIKE fallback path.
    const er = await env.META_DB
      .prepare(
        `SELECT e.id, e.document_id, e.heading, e.content, d.title
           FROM entries e
           JOIN documents d ON d.id = e.document_id
          WHERE e.content LIKE ? OR e.heading LIKE ?
          LIMIT ?`,
      )
      .bind(like, like, limit)
      .all<EntrySearchRow>();
    for (const row of er.results ?? []) {
      hits.push({
        kind: "entry",
        documentId: row.document_id,
        entryId: row.id,
        title: row.title ?? "Untitled",
        snippet: (row.heading || row.content || "").slice(0, 280),
        score: 0.5,
      });
    }
  }

  return jsonResponse({ hits }, {}, env, request);
}

async function handleSearchSemantic(request: Request, env: Env): Promise<Response> {
  const body = (await request.json()) as { query?: string; limit?: number };
  const query = (body.query ?? "").trim();
  const limit = Math.min(50, Math.max(1, body.limit ?? 12));
  if (!query) return jsonResponse({ hits: [] }, {}, env, request);

  const queryVector = await embedQuery(env, query);
  const result = await env.VECTORS.query(queryVector, { topK: limit, returnMetadata: true });
  const matches = result.matches ?? [];

  // Hydrate snippets + titles from D1 in one SELECT.
  const docIds = Array.from(
    new Set(
      matches.map((mt) => {
        const meta = (mt.metadata ?? {}) as Partial<VectorChunkMetadata>;
        return String(meta.docId ?? "");
      }),
    ),
  ).filter(Boolean);
  const titleMap = new Map<string, string>();
  if (docIds.length) {
    const placeholders = docIds.map(() => "?").join(",");
    const rows = await env.META_DB
      .prepare(`SELECT id, title FROM documents WHERE id IN (${placeholders})`)
      .bind(...docIds)
      .all<{ id: string; title: string | null }>();
    for (const r of rows.results ?? []) titleMap.set(r.id, r.title ?? "Untitled");
  }

  const hits = matches.map((mt) => {
    const meta = (mt.metadata ?? {}) as Partial<VectorChunkMetadata>;
    const docId = String(meta.docId ?? "");
    return {
      documentId: docId,
      chunkId: String(meta.chunkId ?? mt.id ?? ""),
      score: mt.score ?? 0,
      snippet: meta.snippet ?? "",
      charStart: meta.charStart,
      charEnd: meta.charEnd,
      title: titleMap.get(docId) ?? "Untitled",
    };
  });

  return jsonResponse({ hits }, {}, env, request);
}

// ---------- Embedding-job parity surface ---------------------------------
//
// These mirror the local indexer's status surface so the adapter contract
// behaves identically on either backend. Cloud indexing is per-call
// synchronous (no background queue), so `queuedRemaining` reflects rows
// in `chunk_embedding_jobs` that are still in a non-terminal state.

async function handleJobsStatus(request: Request, env: Env): Promise<Response> {
  // Counts only non-terminal rows for queuedRemaining. Cloud indexing is
  // per-call synchronous, so a "running" row means a request is mid-flight
  // right now (rare, but possible if two PUTs ran in parallel).
  const queuedRow = await env.META_DB
    .prepare("SELECT COUNT(*) AS n FROM chunk_embedding_jobs WHERE status IN ('queued','running')")
    .first<{ n: number }>();
  const failedRow = await env.META_DB
    .prepare("SELECT COUNT(*) AS n FROM chunk_embedding_jobs WHERE status = 'failed'")
    .first<{ n: number }>();
  const runningRow = await env.META_DB
    .prepare("SELECT document_id FROM chunk_embedding_jobs WHERE status = 'running' ORDER BY updated_at DESC LIMIT 1")
    .first<{ document_id: string }>();
  const queued = Number(queuedRow?.n ?? 0);
  const failed = Number(failedRow?.n ?? 0);
  const phase: "running" | "error" | "idle" =
    runningRow ? "running" : failed > 0 ? "error" : "idle";
  return jsonResponse(
    {
      phase,
      currentDocId: runningRow?.document_id ?? undefined,
      queuedRemaining: queued,
      failedCount: failed,
      reason: phase === "error" ? `${failed} job(s) failed` : undefined,
    },
    {}, env, request,
  );
}

async function handleClearJobs(request: Request, env: Env): Promise<Response> {
  await env.META_DB.prepare("DELETE FROM chunk_embedding_jobs").run();
  return jsonResponse({ ok: true }, {}, env, request);
}

// ---------- Import-jobs CRUD (IDB parity) --------------------------------
//
// Mirrors the local `importJobs` IDB store row-for-row. The mirror runner
// walks listImportJobs() locally and upserts each row here so another tab
// (or a future fresh-tab restore) can see the same in-flight import set.

async function handleListImportJobs(request: Request, env: Env): Promise<Response> {
  const r = await env.META_DB
    .prepare("SELECT * FROM import_jobs ORDER BY updated_at DESC")
    .all<ImportJobRow>();
  return jsonResponse(
    { importJobs: (r.results ?? []).map(rowToImportJob) },
    {}, env, request,
  );
}

async function handleGetImportJob(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const row = await env.META_DB
    .prepare("SELECT * FROM import_jobs WHERE id = ?")
    .bind(id)
    .first<ImportJobRow>();
  if (!row) return errorResponse(404, `Import job ${id} not found.`, env, request);
  return jsonResponse({ importJob: rowToImportJob(row) }, {}, env, request);
}

async function handlePutImportJob(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const body = (await request.json()) as { importJob?: InboundImportJobBody };
  const j = body.importJob;
  if (!j || j.id !== id) {
    return errorResponse(400, "Body must include `importJob` with matching id.", env, request);
  }
  await env.META_DB.prepare(
    `INSERT INTO import_jobs (id, file_name, file_size, detected_source_type, status,
         progress, current_step_label, documents_processed, entries_processed,
         document_ids, error, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
         file_name=excluded.file_name,
         file_size=excluded.file_size,
         detected_source_type=excluded.detected_source_type,
         status=excluded.status,
         progress=excluded.progress,
         current_step_label=excluded.current_step_label,
         documents_processed=excluded.documents_processed,
         entries_processed=excluded.entries_processed,
         document_ids=excluded.document_ids,
         error=excluded.error,
         updated_at=excluded.updated_at`,
  )
    .bind(
      String(j.id),
      String(j.fileName ?? ""),
      Number(j.fileSize ?? 0),
      j.detectedSourceType ?? null,
      String(j.status ?? "pending"),
      Number(j.progress ?? 0),
      j.currentStepLabel ?? null,
      Number(j.documentsProcessed ?? 0),
      Number(j.entriesProcessed ?? 0),
      j.documentIds ? JSON.stringify(j.documentIds) : null,
      j.error ?? null,
      String(j.createdAt ?? new Date().toISOString()),
      String(j.updatedAt ?? new Date().toISOString()),
    )
    .run();
  return jsonResponse({ ok: true }, {}, env, request);
}

async function handleDeleteImportJob(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  await env.META_DB.prepare("DELETE FROM import_jobs WHERE id = ?").bind(id).run();
  return jsonResponse({ ok: true }, {}, env, request);
}

async function handleCoverage(request: Request, env: Env): Promise<Response> {
  const totalDocsRow = await env.META_DB
    .prepare("SELECT COUNT(*) AS n FROM documents WHERE deleted_at IS NULL")
    .first<{ n: number }>();
  const coveredRow = await env.META_DB
    .prepare("SELECT COUNT(DISTINCT document_id) AS n FROM chunk_vectors")
    .first<{ n: number }>();
  const chunksRow = await env.META_DB
    .prepare("SELECT COUNT(*) AS n FROM chunk_vectors")
    .first<{ n: number }>();
  return jsonResponse(
    {
      totalDocs: Number(totalDocsRow?.n ?? 0),
      coveredDocs: Number(coveredRow?.n ?? 0),
      totalChunks: Number(chunksRow?.n ?? 0),
    },
    {}, env, request,
  );
}

// ---------- Indexing pipeline --------------------------------------------

/** Chunk size + overlap chosen to match the local `content-map` segment
 *  granularity (~1k chars per segment). Per the plan §"Mirror — Re-index":
 *  vectors aren't byte-comparable across models but the chunk boundaries
 *  should roughly track so user-visible hit grain matches the local UX. */
const CHUNK_MAX_CHARS = 1200;
const CHUNK_OVERLAP_CHARS = 150;

function chunkText(text: string, maxChars = CHUNK_MAX_CHARS, overlap = CHUNK_OVERLAP_CHARS): Array<{ text: string; start: number; end: number }> {
  const out: Array<{ text: string; start: number; end: number }> = [];
  if (!text) return out;
  let i = 0;
  while (i < text.length) {
    const end = Math.min(text.length, i + maxChars);
    out.push({ text: text.slice(i, end), start: i, end });
    if (end === text.length) break;
    i = Math.max(0, end - overlap);
  }
  return out;
}

async function embedQuery(env: Env, query: string): Promise<number[]> {
  const model = env.EMBEDDING_MODEL || "@cf/baai/bge-small-en-v1.5";
  // BGE wants the "Represent this sentence…" prefix on queries to match
  // the local LocalEmbeddingsAdapter behaviour.
  const prefixed = `Represent this sentence for searching relevant passages: ${query}`;
  const ai = env.AI as unknown as WorkersAiClient;
  const r = await ai.run(model, { text: [prefixed] });
  return r.data[0];
}

async function embedPassages(env: Env, texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const model = env.EMBEDDING_MODEL || "@cf/baai/bge-small-en-v1.5";
  // Workers AI caps batch size per call — chunk into groups of 32.
  const BATCH = 32;
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH);
    const ai = env.AI as unknown as WorkersAiClient;
    const r = await ai.run(model, { text: batch });
    if (!Array.isArray(r?.data)) {
      throw new Error(`Workers AI returned no data (batch ${i / BATCH}).`);
    }
    for (const v of r.data) out.push(v);
  }
  return out;
}

async function handleIndexDoc(request: Request, env: Env, _ctx: ExecutionContext, m: RegExpMatchArray): Promise<Response> {
  const id = decodeURIComponent(m[1]);
  const errors: string[] = [];
  let chunksIndexed = 0;
  let chunksFailed = 0;

  // Preferred source: content_map (one row per coherent segment).
  // Fallback: entries concatenated then chunked.
  let segments: Array<{ chunkId: string; text: string; charStart: number; charEnd: number }> = [];
  const cmRows = await env.META_DB
    .prepare("SELECT id, index_text, content_head FROM content_map WHERE document_id = ? ORDER BY segment_number ASC")
    .bind(id)
    .all<{ id: string; index_text: string; content_head: string }>();
  if ((cmRows.results ?? []).length > 0) {
    let cursor = 0;
    for (const r of cmRows.results!) {
      const text = (r.index_text || r.content_head || "").trim();
      if (!text) continue;
      const start = cursor;
      const end = cursor + text.length;
      segments.push({ chunkId: r.id, text, charStart: start, charEnd: end });
      cursor = end;
    }
  } else {
    const mdRow = await env.META_DB
      .prepare("SELECT markdown FROM markdown WHERE document_id = ?")
      .bind(id)
      .first<{ markdown: string }>();
    const md = mdRow?.markdown ?? "";
    if (md) {
      const chunks = chunkText(md);
      segments = chunks.map((c, i) => ({
        chunkId: `chunk-${i.toString().padStart(5, "0")}`,
        text: c.text,
        charStart: c.start,
        charEnd: c.end,
      }));
    } else {
      // Last-resort fallback: concatenate entries in order and chunk.
      // Covers docs where the importer wrote entries but neither
      // content_map nor markdown — common for entry-only sources.
      const entryRows = await env.META_DB
        .prepare(
          "SELECT id, heading, content FROM entries WHERE document_id = ? ORDER BY order_index ASC",
        )
        .bind(id)
        .all<{ id: string; heading: string | null; content: string | null }>();
      const combined = (entryRows.results ?? [])
        .map((e) => [e.heading, e.content].filter(Boolean).join("\n"))
        .filter((s) => s.trim().length > 0)
        .join("\n\n");
      if (combined) {
        const chunks = chunkText(combined);
        segments = chunks.map((c, i) => ({
          chunkId: `entry-chunk-${i.toString().padStart(5, "0")}`,
          text: c.text,
          charStart: c.start,
          charEnd: c.end,
        }));
      }
    }
  }

  // Load the prior vector ID set BEFORE we start so we can compute the
  // superseded set after upsert and remove orphans from Vectorize.
  // This is what keeps `chunk_vectors` a deterministic source of truth
  // for the delete-cascade path — if we skipped this, a re-index that
  // changed chunk IDs would silently strand vectors in the index.
  const priorRows = await env.META_DB
    .prepare("SELECT vector_id FROM chunk_vectors WHERE document_id = ?")
    .bind(id)
    .all<{ vector_id: string }>();
  const priorVectorIds = new Set((priorRows.results ?? []).map((r) => r.vector_id));

  if (segments.length === 0) {
    // No content to index — purge stale vectors first, then chunk_vectors
    // rows ONLY on successful Vectorize delete. Same tombstone strategy
    // as the superseded-purge path: keep the source-of-truth row set
    // intact on failure so a retry can complete cleanup deterministically.
    if (priorVectorIds.size > 0) {
      try {
        await env.VECTORS.deleteByIds(Array.from(priorVectorIds));
        await env.META_DB
          .prepare("DELETE FROM chunk_vectors WHERE document_id = ?")
          .bind(id)
          .run();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.warn(`[index] Vectorize purge for empty ${id} failed:`, e);
        return jsonResponse(
          {
            indexStatus: "partial",
            chunksIndexed: 0,
            chunksFailed: 0,
            errors: [
              "No content to index.",
              `Vectorize purge failed: ${msg}. chunk_vectors retained for retry.`,
            ],
          },
          {}, env, request,
        );
      }
    }
    return jsonResponse(
      { indexStatus: "complete", chunksIndexed: 0, chunksFailed: 0, errors: ["No content to index."] },
      {}, env, request,
    );
  }

  // Mark job running.
  const nowIso = new Date().toISOString();
  await env.META_DB
    .prepare(
      `INSERT INTO chunk_embedding_jobs (id, document_id, status, attempts, enqueued_at, updated_at)
       VALUES (?, ?, 'running', 1, ?, ?)
       ON CONFLICT(id) DO UPDATE SET status='running', attempts=chunk_embedding_jobs.attempts+1, updated_at=excluded.updated_at`,
    )
    .bind(id, id, nowIso, nowIso)
    .run();

  // Embed in batches. A batch-level failure is recorded against every
  // segment in that batch but does not abort the rest of the run.
  const BATCH = 32;
  const model = env.EMBEDDING_MODEL || "@cf/baai/bge-small-en-v1.5";
  type ToUpsert = { id: string; values: number[]; metadata: VectorChunkMetadata };
  const upserts: ToUpsert[] = [];
  for (let i = 0; i < segments.length; i += BATCH) {
    const batch = segments.slice(i, i + BATCH);
    try {
      const vectors = await embedPassages(env, batch.map((s) => s.text));
      batch.forEach((s, j) => {
        const vec = vectors[j];
        if (!vec || vec.length === 0) {
          chunksFailed++;
          errors.push(`Empty vector for chunk ${s.chunkId}`);
          return;
        }
        upserts.push({
          id: `${id}:${s.chunkId}`,
          values: vec,
          metadata: {
            docId: id,
            chunkId: s.chunkId,
            snippet: s.text.slice(0, 280),
            charStart: s.charStart,
            charEnd: s.charEnd,
          },
        });
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      chunksFailed += batch.length;
      errors.push(`Embed batch starting at ${i} failed: ${msg}`);
    }
  }

  // Upsert in batches; Vectorize caps per-call size too. We track the
  // EXACT upserts that succeeded (not just a count) so chunk_vectors
  // reflects what's actually in Vectorize even on partial failure.
  const VBATCH = 100;
  const succeeded: ToUpsert[] = [];
  for (let i = 0; i < upserts.length; i += VBATCH) {
    const slice = upserts.slice(i, i + VBATCH);
    try {
      await env.VECTORS.upsert(slice);
      for (const u of slice) succeeded.push(u);
      chunksIndexed += slice.length;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      chunksFailed += slice.length;
      errors.push(`Vectorize upsert at ${i} failed: ${msg}`);
    }
  }

  // Compute the superseded vector IDs — old IDs from before this run that
  // were NOT re-written by a successful upsert this run. We must purge
  // them from Vectorize, otherwise the chunk_vectors source-of-truth
  // (used by DELETE /docs/:id) will no longer cover everything in the
  // index for this doc, causing orphan vectors to survive cascade-delete.
  const succeededIds = new Set(succeeded.map((u) => u.id));
  const supersededIds: string[] = [];
  for (const old of priorVectorIds) if (!succeededIds.has(old)) supersededIds.push(old);
  let supersededPurged = true;
  if (supersededIds.length > 0) {
    try {
      await env.VECTORS.deleteByIds(supersededIds);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      errors.push(`Vectorize orphan purge failed: ${msg}`);
      supersededPurged = false;
    }
  }

  // chunk_vectors lifecycle: only delete the rows whose Vectorize
  // counterparts we successfully purged. If superseded delete failed,
  // we KEEP those prior rows as a tombstone set so a future re-index
  // (or DELETE /docs/:id) can retry the Vectorize cleanup
  // deterministically. Then upsert the rows for this run's successes.
  const removeRowsBinding: D1PreparedStatement = supersededPurged
    ? env.META_DB.prepare("DELETE FROM chunk_vectors WHERE document_id = ?").bind(id)
    : env.META_DB
      .prepare(
        `DELETE FROM chunk_vectors WHERE document_id = ? AND vector_id IN (${succeeded.map(() => "?").join(",") || "''"})`,
      )
      .bind(id, ...succeeded.map((u) => u.id));
  await env.META_DB.batch([
    removeRowsBinding,
    ...succeeded.map((u) =>
      env.META_DB
        .prepare(
          `INSERT INTO chunk_vectors (document_id, chunk_id, vector_id, model_id, embedded_at)
           VALUES (?, ?, ?, ?, ?)`,
        )
        .bind(id, (u.metadata as VectorChunkMetadata).chunkId, u.id, model, nowIso),
    ),
  ]);

  const indexStatus: "complete" | "partial" | "failed" =
    chunksFailed === 0 && chunksIndexed > 0
      ? "complete"
      : chunksIndexed === 0
        ? "failed"
        : "partial";

  // Final job state — `complete`, `partial`, or `failed`. Never write
  // back `queued` here (the old code did, which made handleJobsStatus
  // report a non-zero queue forever after a successful single-doc index
  // and showed misleading "still indexing" UX). Terminal status only.
  await env.META_DB
    .prepare(
      `UPDATE chunk_embedding_jobs SET status = ?, last_error = ?, updated_at = ? WHERE id = ?`,
    )
    .bind(indexStatus, errors[0] ?? null, new Date().toISOString(), id)
    .run();

  return jsonResponse(
    { indexStatus, chunksIndexed, chunksFailed, errors },
    {},
    env,
    request,
  );
}
