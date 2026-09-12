#!/usr/bin/env node
/**
 * Copy chapters/chN-*.md (and optional chN.assessments.json) into constants.ts.
 * Do not hand-edit constants.ts.
 *
 *   node scripts/inject-chapter.mjs ch1
 */
import fs from 'node:fs';
import path from 'node:path';

const id = process.argv[2];
if (!id || !/^ch\d+$/.test(id)) {
  console.error('usage: node scripts/inject-chapter.mjs ch1');
  process.exit(1);
}

const chapterDir = 'chapters';
const mdFiles = fs
  .readdirSync(chapterDir)
  .filter((f) => f.startsWith(`${id}-`) && f.endsWith('.md'));
if (mdFiles.length !== 1) {
  console.error(`expected exactly one ${chapterDir}/${id}-*.md, found: ${mdFiles.join(', ') || '(none)'}`);
  process.exit(1);
}

const md = fs.readFileSync(path.join(chapterDir, mdFiles[0]), 'utf8');
const title = (md.match(/^#\s+(.+)$/m) || [])[1]?.trim();
if (!title) {
  console.error(`${mdFiles[0]} needs an H1 title`);
  process.exit(1);
}

const escaped = md.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${');

const assessPath = path.join(chapterDir, `${id}.assessments.json`);
const assessments = fs.existsSync(assessPath)
  ? JSON.parse(fs.readFileSync(assessPath, 'utf8'))
  : null;

const srcPath = 'constants.ts';
const src = fs.readFileSync(srcPath, 'utf8');
const idNeedle = `id: '${id}'`;
const idAt = src.indexOf(idNeedle);
if (idAt < 0) {
  console.error(`${id} is not in ${srcPath}`);
  process.exit(1);
}

const objStart = src.lastIndexOf('\n  {', idAt);
const nextObj = src.indexOf('\n  {\n    id:', idAt + 1);
const objEnd = nextObj >= 0 ? nextObj : src.indexOf('\n];', idAt);
if (objStart < 0 || objEnd < 0) {
  console.error(`could not bound the ${id} object in ${srcPath}`);
  process.exit(1);
}

let block = src.slice(objStart, objEnd);
block = block.replace(/title:\s*"[^"]*"/, `title: ${JSON.stringify(title)}`);
block = replaceTemplate(block, 'content', escaped);

if (assessments?.quizzes) {
  block = replaceJsonArray(block, 'quizzes', assessments.quizzes);
}
if (assessments?.flashcards) {
  block = replaceJsonArray(block, 'flashcards', assessments.flashcards);
}

fs.writeFileSync(srcPath, src.slice(0, objStart) + block + src.slice(objEnd));
console.log(`injected ${id} ← ${mdFiles[0]}${assessments ? ` + ${id}.assessments.json` : ''}`);

function replaceTemplate(block, field, escapedValue) {
  const marker = `${field}: \``;
  const i = block.indexOf(marker);
  if (i < 0) throw new Error(`no ${field} template in ${id}`);
  let j = i + marker.length;
  while (j < block.length) {
    if (block[j] === '\\') {
      j += 2;
      continue;
    }
    if (block[j] === '`') break;
    j += 1;
  }
  if (j >= block.length) throw new Error(`unclosed ${field} template in ${id}`);
  return `${block.slice(0, i)}${marker}${escapedValue}\`${block.slice(j + 1)}`;
}

function replaceJsonArray(block, field, value) {
  const marker = `${field}: [`;
  const i = block.indexOf(marker);
  if (i < 0) throw new Error(`no ${field} array in ${id}`);
  const start = i + `${field}: `.length;
  const end = matchBracket(block, start);
  const json = JSON.stringify(value, null, 6)
    .replace(/\n/g, '\n      ')
    .replace(/^\[\n      /, '[\n      ')
    .replace(/\n      \]$/, '\n    ]');
  return block.slice(0, start) + json + block.slice(end + 1);
}

function matchBracket(s, openIdx) {
  let depth = 0;
  for (let i = openIdx; i < s.length; i += 1) {
    const c = s[i];
    if (c === '[' || c === '{') depth += 1;
    else if (c === ']' || c === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  throw new Error('unbalanced brackets');
}
