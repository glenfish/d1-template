-- Task #227 — D1 schema for the single-user cloud vault.
--
-- Mirrors the IndexedDB stores in client/src/lib/vault/types.ts (STORE_NAMES)
-- 1:1 so the adapter does pure row marshalling, not semantic translation.
--
-- Column shapes match the Local* types in client/src/lib/vault/types.ts:
--   - documents          ↔ LocalSourceDocument
--   - entries            ↔ LocalContentEntry
--   - markdown           ↔ LocalMarkdownDocument
--   - content_map        ↔ LocalContentMapEntry
--   - document_index     ↔ LocalDocumentIndex
--   - import_jobs        ↔ LocalImportJob
--   - chunk_embedding_jobs ↔ EmbeddingJobRow (status surface, not vector store)
--   - chunk_vectors      ↔ NEW — deterministic source of truth for which
--                          Vectorize IDs belong to which doc, so cascade-
--                          delete on DELETE /docs/:id never relies on a
--                          metadata-filter scan.
--
-- Single-user, single-tenant: NO `userId`/`orgId` columns. Task #198 will
-- add those additively.

CREATE TABLE IF NOT EXISTS documents (
  id                   TEXT PRIMARY KEY,
  source_type          TEXT NOT NULL,
  source_import_id     TEXT NOT NULL,
  source_document_id   TEXT,
  title                TEXT,
  created_at           TEXT,
  updated_at           TEXT,
  imported_at          TEXT NOT NULL,
  entry_count          INTEGER NOT NULL DEFAULT 0,
  checksum             TEXT NOT NULL,
  size_bytes           INTEGER,
  parser_version       TEXT NOT NULL,
  index_version        TEXT,
  reindex_required     INTEGER NOT NULL DEFAULT 0,
  deleted_at           TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_imported_at ON documents(imported_at);
CREATE INDEX IF NOT EXISTS idx_documents_checksum    ON documents(checksum);

CREATE TABLE IF NOT EXISTS entries (
  id                    TEXT PRIMARY KEY,
  document_id           TEXT NOT NULL,
  role                  TEXT,
  content               TEXT NOT NULL,
  heading               TEXT,
  heading_level         INTEGER,
  order_index           INTEGER NOT NULL,
  markdown_start_offset INTEGER NOT NULL,
  markdown_end_offset   INTEGER NOT NULL,
  offset_encoding       TEXT NOT NULL DEFAULT 'utf16',
  created_at            TEXT,
  metadata              TEXT  -- JSON
);
CREATE INDEX IF NOT EXISTS idx_entries_document_id  ON entries(document_id);
CREATE INDEX IF NOT EXISTS idx_entries_doc_order    ON entries(document_id, order_index);

CREATE TABLE IF NOT EXISTS markdown (
  document_id  TEXT PRIMARY KEY,
  markdown     TEXT NOT NULL,
  generated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS content_map (
  id                  TEXT PRIMARY KEY,
  document_id         TEXT NOT NULL,
  segment_number      INTEGER NOT NULL,
  entry_ids           TEXT NOT NULL,   -- JSON array of entry ids
  heading_preview     TEXT,
  content_head        TEXT NOT NULL,
  content_tail        TEXT,
  response_head       TEXT,
  response_tail       TEXT,
  keywords            TEXT NOT NULL,   -- JSON array of strings
  entry_start_index   INTEGER NOT NULL,
  entry_end_index     INTEGER NOT NULL,
  index_text          TEXT NOT NULL,
  anchor              TEXT NOT NULL,   -- JSON ContentAnchor
  index_version       TEXT NOT NULL,
  created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_content_map_document_id ON content_map(document_id);
CREATE INDEX IF NOT EXISTS idx_content_map_doc_seg     ON content_map(document_id, segment_number);

CREATE TABLE IF NOT EXISTS document_index (
  document_id          TEXT PRIMARY KEY,
  source_type          TEXT NOT NULL,
  title                TEXT,
  created_at           TEXT,
  updated_at           TEXT,
  entry_count          INTEGER NOT NULL,
  segment_count        INTEGER NOT NULL,
  keyword_fingerprint  TEXT NOT NULL,  -- JSON array of strings
  index_text_preview   TEXT NOT NULL,
  index_version        TEXT NOT NULL,
  generated_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_jobs (
  id                    TEXT PRIMARY KEY,
  file_name             TEXT NOT NULL,
  file_size             INTEGER NOT NULL,
  detected_source_type  TEXT,
  status                TEXT NOT NULL,
  progress              INTEGER NOT NULL DEFAULT 0,
  current_step_label    TEXT,
  documents_processed   INTEGER NOT NULL DEFAULT 0,
  entries_processed     INTEGER NOT NULL DEFAULT 0,
  document_ids          TEXT,    -- JSON array
  error                 TEXT,
  created_at            TEXT NOT NULL,
  updated_at            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk_embedding_jobs (
  id           TEXT PRIMARY KEY,            -- e.g. `${docId}`
  document_id  TEXT NOT NULL,
  status       TEXT NOT NULL,               -- queued | running | failed
  attempts     INTEGER NOT NULL DEFAULT 0,
  last_error   TEXT,
  enqueued_at  TEXT NOT NULL,
  updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunk_jobs_status ON chunk_embedding_jobs(status);

-- Source of truth for which Vectorize IDs belong to which doc.
-- DELETE /docs/:id enumerates these and passes them to Vectorize
-- `deleteByIds` — no reliance on metadata-filter scans.
CREATE TABLE IF NOT EXISTS chunk_vectors (
  document_id  TEXT NOT NULL,
  chunk_id     TEXT NOT NULL,
  vector_id    TEXT NOT NULL,            -- "<docId>:<chunkId>" — what Vectorize stores
  model_id     TEXT NOT NULL,
  embedded_at  TEXT NOT NULL,
  PRIMARY KEY (document_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_vectors_doc ON chunk_vectors(document_id);

-- FTS5 virtual tables for fast keyword search over entries + content_map.
-- D1 supports FTS5 in modern compatibility dates; if the deploy errors out
-- on these statements, drop them and the Worker will fall back to LIKE.
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
  content,
  heading,
  document_id UNINDEXED,
  entry_id    UNINDEXED,
  content='entries',
  content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS content_map_fts USING fts5(
  index_text,
  heading_preview,
  keywords,
  document_id  UNINDEXED,
  segment_id   UNINDEXED,
  content='content_map',
  content_rowid='rowid'
);

-- Keep the FTS shadow in sync. AFTER triggers fire automatically on writes
-- so the Worker doesn't need a separate "rebuild FTS" path.
CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
  INSERT INTO entries_fts(rowid, content, heading, document_id, entry_id)
    VALUES (new.rowid, new.content, COALESCE(new.heading, ''), new.document_id, new.id);
END;
CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
  INSERT INTO entries_fts(entries_fts, rowid, content, heading, document_id, entry_id)
    VALUES('delete', old.rowid, old.content, COALESCE(old.heading, ''), old.document_id, old.id);
END;
CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
  INSERT INTO entries_fts(entries_fts, rowid, content, heading, document_id, entry_id)
    VALUES('delete', old.rowid, old.content, COALESCE(old.heading, ''), old.document_id, old.id);
  INSERT INTO entries_fts(rowid, content, heading, document_id, entry_id)
    VALUES (new.rowid, new.content, COALESCE(new.heading, ''), new.document_id, new.id);
END;

CREATE TRIGGER IF NOT EXISTS content_map_ai AFTER INSERT ON content_map BEGIN
  INSERT INTO content_map_fts(rowid, index_text, heading_preview, keywords, document_id, segment_id)
    VALUES (new.rowid, new.index_text, COALESCE(new.heading_preview, ''), new.keywords, new.document_id, new.id);
END;
CREATE TRIGGER IF NOT EXISTS content_map_ad AFTER DELETE ON content_map BEGIN
  INSERT INTO content_map_fts(content_map_fts, rowid, index_text, heading_preview, keywords, document_id, segment_id)
    VALUES('delete', old.rowid, old.index_text, COALESCE(old.heading_preview, ''), old.keywords, old.document_id, old.id);
END;
CREATE TRIGGER IF NOT EXISTS content_map_au AFTER UPDATE ON content_map BEGIN
  INSERT INTO content_map_fts(content_map_fts, rowid, index_text, heading_preview, keywords, document_id, segment_id)
    VALUES('delete', old.rowid, old.index_text, COALESCE(old.heading_preview, ''), old.keywords, old.document_id, old.id);
  INSERT INTO content_map_fts(rowid, index_text, heading_preview, keywords, document_id, segment_id)
    VALUES (new.rowid, new.index_text, COALESCE(new.heading_preview, ''), new.keywords, new.document_id, new.id);
END;
