/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const docsRoot = path.resolve(here, '../../..')
const repoRoot = path.resolve(docsRoot, '..')
const releaseNotesPath = path.join(docsRoot, 'zh/release-notes.md')
const changelogPath = path.join(repoRoot, 'CHANGELOG.md')

export interface ReleaseEntry {
  version: string
  date: string
  href: string
  title: string
  desc: string
}

export interface OperatorEntry {
  date: string
  time: string
  title: string
  desc: string
}

function stripMarkdown(text: string): string {
  return text
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/<[^>]+>/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function splitSummary(summary: string): { title: string; desc: string } {
  const [first, ...rest] = summary.split('，')
  return {
    title: first.trim(),
    desc: rest.join('，').trim() || summary,
  }
}

function parseReleases(): ReleaseEntry[] {
  let raw: string
  try {
    raw = readFileSync(releaseNotesPath, 'utf8')
  } catch {
    return []
  }

  const entries: ReleaseEntry[] = []
  const sectionRe =
    /<section class="release-entry"[\s\S]*?<time datetime="([^"]+)">[\s\S]*?## \[([^\]]+)\]\(([^)]+)\)[^\n]*\n([\s\S]*?)\s*\[查看版本说明\]/g

  let match: RegExpExecArray | null
  while ((match = sectionRe.exec(raw))) {
    const [, date, version, href, summaryRaw] = match
    const summary = stripMarkdown(summaryRaw)
    const { title, desc } = splitSummary(summary)
    entries.push({
      version: version.trim(),
      date,
      href: href.replace(/^\.\/release-notes\//, '/zh/release-notes/').replace(/\.md$/, '.html'),
      title,
      desc,
    })
  }

  return entries.sort((a, b) => b.date.localeCompare(a.date))
}

function parseOperators(): OperatorEntry[] {
  let raw: string
  try {
    raw = readFileSync(changelogPath, 'utf8')
  } catch {
    return []
  }

  const entries: OperatorEntry[] = []
  let currentDate = ''

  for (const line of raw.split(/\r?\n/)) {
    const dateMatch = line.match(/^### 【(\d{4}-\d{2}-\d{2})】/)
    if (dateMatch) {
      currentDate = dateMatch[1]
      continue
    }

    const itemMatch = line.match(/^- 【([^】]+)】(.+)$/)
    if (!itemMatch || !currentDate) continue
    if (entries.some((entry) => entry.date === currentDate)) continue

    const text = stripMarkdown(itemMatch[2])
    const colon = text.indexOf('：')
    let title: string
    let desc: string

    if (colon > 0) {
      title = text.slice(0, colon).trim()
      desc = text.slice(colon + 1).trim()
    } else {
      const sentenceEnd = text.search(/[。；]/)
      title = (sentenceEnd > 0 ? text.slice(0, sentenceEnd) : text).slice(0, 28)
      desc = sentenceEnd > 0 ? text.slice(sentenceEnd + 1).trim() : ''
    }

    if (!title) title = itemMatch[1].split(/\s*\/\s*/)[0].trim()
    if (desc.length > 90) desc = `${desc.slice(0, 90).trimEnd()}…`

    const [, month, day] = currentDate.split('-')
    entries.push({ date: currentDate, time: `${month}.${day}`, title, desc })
  }

  return entries.sort((a, b) => b.date.localeCompare(a.date))
}

export default {
  load() {
    return {
      releases: parseReleases(),
      operators: parseOperators(),
    }
  },
}
