const REF_TOKEN_RE = /\[\[(R\d+)\]\]/g;
const REF_LINE_RE =
  /^(?:参考依据|依据来源|引用依据|依据|参考|来源|references?|citations?|sources?)\s*[:：]?\s*(.+)$/i;
const CITATION_COLUMN_RE =
  /^(?:依据|参考|来源|引用|reference|references|citation|citations|source|sources)$/i;

export function buildContentBlocks(text) {
  const blocks = [];
  const lines = String(text || "").split("\n");
  let buffer = [];
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const referenceRefs = referenceLineRefs(line);
    if (referenceRefs.length) {
      flushTextBlock(blocks, buffer);
      buffer = [];
      attachRefs(blocks, referenceRefs);
      continue;
    }
    if (isTableLine(line)) {
      flushTextBlock(blocks, buffer);
      buffer = [];
      const tableLines = collectTableLines(lines, index);
      index += tableLines.length - 1;
      addTableBlock(blocks, tableLines);
      continue;
    }
    const heading = standaloneHeading(line);
    if (heading) {
      flushTextBlock(blocks, buffer);
      buffer = [];
      blocks.push({ type: "heading", text: heading, refs: [] });
      continue;
    }
    buffer.push(line);
  }
  flushTextBlock(blocks, buffer);
  return blocks;
}

export function visibleCitationIds(text) {
  const seen = new Set();
  const ids = [];
  buildContentBlocks(text).forEach((block) => collectVisibleRefs(block, seen, ids));
  return ids;
}

export function referenceMap(references) {
  const map = {};
  (references || []).forEach((ref) => {
    if (ref?.ref_id) map[ref.ref_id] = ref;
  });
  return map;
}

export function citationMeta(ref) {
  const parts = [];
  if (ref?.source_clause) parts.push(`条款 ${ref.source_clause}`);
  if (ref?.source_doc) parts.push(ref.source_doc);
  if (ref?.source_pages) parts.push(`第 ${ref.source_pages} 页`);
  return parts.join(" · ");
}

export function splitRefTokens(text) {
  const parts = [];
  const raw = String(text || "");
  let lastIndex = 0;
  for (const match of raw.matchAll(REF_TOKEN_RE)) {
    if (match.index > lastIndex) {
      parts.push({ text: raw.slice(lastIndex, match.index) });
    }
    parts.push({ refId: match[1] });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < raw.length) parts.push({ text: raw.slice(lastIndex) });
  return parts;
}

function collectTableLines(lines, startIndex) {
  const tableLines = [];
  let index = startIndex;
  while (index < lines.length && isTableLine(lines[index])) {
    tableLines.push(lines[index]);
    index += 1;
  }
  return tableLines;
}

function flushTextBlock(blocks, buffer) {
  if (!buffer.length) return;
  const text = buffer.join("\n").replace(/^\n+|\n+$/g, "");
  if (text.trim()) blocks.push({ type: "text", text, refs: [] });
}

function addTableBlock(blocks, tableLines) {
  const block = tableBlock(tableLines);
  if (block.refs.length && attachRefs(blocks, block.refs)) block.refs = [];
  blocks.push(block);
}

function tableBlock(lines) {
  let rows = lines.map(parseTableCells).filter((cells) => !isSeparatorCells(cells));
  const citationIndex = citationColumnIndex(rows[0] || []);
  const refs = commonRowRefs(rows, citationIndex);
  if (refs.length) rows = rows.map((row) => row.filter((_, index) => index !== citationIndex));
  return { type: "table", rows, refs };
}

function commonRowRefs(rows, citationIndex) {
  if (citationIndex < 0 || rows.length < 2) return [];
  const refsByRow = rows.slice(1).map((row) => extractRefs(row[citationIndex] || ""));
  if (!refsByRow.length || refsByRow.some((refs) => !refs.length)) return [];
  const first = refKey(refsByRow[0]);
  return refsByRow.every((refs) => refKey(refs) === first) ? refsByRow[0] : [];
}

function attachRefs(blocks, refs) {
  const last = blocks[blocks.length - 1];
  if (!last) return false;
  last.refs = uniqueRefs([...(last.refs || []), ...refs]);
  return true;
}

function collectVisibleRefs(block, seen, ids) {
  blockRefs(block).forEach((refId) => {
    if (seen.has(refId)) return;
    seen.add(refId);
    ids.push(refId);
  });
}

function blockRefs(block) {
  if (block.type === "table") {
    return [...(block.refs || []), ...block.rows.flatMap((row) => row.flatMap(extractRefs))];
  }
  return [...extractRefs(block.text), ...(block.refs || [])];
}

function extractRefs(text) {
  return uniqueRefs([...String(text || "").matchAll(REF_TOKEN_RE)].map((match) => match[1]));
}

function referenceLineRefs(line) {
  const match = REF_LINE_RE.exec(String(line || "").trim());
  if (!match) return [];
  const refs = extractRefs(match[1]);
  return match[1].replace(REF_TOKEN_RE, "").trim() ? [] : refs;
}

function standaloneHeading(line) {
  const match = /^\*\*([^*]+)\*\*$/.exec(String(line || "").trim());
  return match ? match[1].trim() : null;
}

function isTableLine(line) {
  const text = String(line || "").trim();
  return text.startsWith("|") && text.endsWith("|") && text.length > 1;
}

function parseTableCells(line) {
  return String(line)
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isSeparatorCells(cells) {
  return cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function citationColumnIndex(header) {
  return header.findIndex((cell) => CITATION_COLUMN_RE.test(String(cell).trim()));
}

function uniqueRefs(refs) {
  return refs.filter((ref, index) => ref && refs.indexOf(ref) === index);
}

function refKey(refs) {
  return uniqueRefs(refs).join("|");
}
