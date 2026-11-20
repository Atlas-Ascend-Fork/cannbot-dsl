/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

<script setup lang="ts">
import { computed, ref } from 'vue'

const props = withDefaults(defineProps<{ lang?: 'zh' | 'en' }>(), { lang: 'zh' })

const copied = ref('')
const pipCommand = 'python -m pip install cannbot-dsl'
const workflowCommand = 'npx @cannbot-plugin/cannbot@latest install ops-direct-invoke'

const zh = {
  subtitle: 'Agent 友好的 Ascend NPU 算子开发 DSL',
  heroCopy: '用 Python 描述数据搬运、片上存储和算子计算，配套编程模型、工程样例与 API 文档。',
  start: '开始使用',
  repo: '源码仓库',
  startHref: '/zh/install/source-build',
  features: [
    { key: 'start', icon: '🔧', title: '开始使用', desc: '完成环境准备，运行已有样例，开始编写第一个算子。', href: '/zh/install/source-build' },
    { key: 'guide', icon: '🚀', title: '入门教程', desc: '从语言机制、数据与硬件到算子实践和工程化，系统理解 DSL。', href: '/zh/guide/overview' },
    { key: 'samples', icon: '🧩', title: '样例导航', desc: '按类别查看算子源码、运行说明和测试方法。', href: 'https://gitcode.com/cann/cannbot-dsl/tree/master/samples', external: true },
    { key: 'api', icon: '⚙️', title: 'API 文档', desc: '查阅 Host、Kernel 和 AI CPU 接口与数据结构。', href: '/zh/api/' },
  ],
  quickstart: '快速接入',
  pip: {
    title: '安装 CANNBot-DSL',
    copy: '安装 Python 包后即可运行 DSL 样例，开始描述数据搬运、片上存储与算子计算。',
    start: '开始安装',
    quick: '查看快速上手 →',
    href: '/zh/install/source-build',
  },
  workflow: {
    title: '安装官方工作流',
    badge: '官方',
    copy: '通过 CANNBot 官方安装器，把算子开发工作流接入当前 AI 编程工具。',
    action: '查看官方工作流 ↗',
  },
  copyText: '复制',
  copied: '已复制',
  copyFailed: '请手动复制',
  example: '核心示例',
  codeBadge: '核心片段',
  codeDoc: '查看文档',
  codeAria: 'Matmul Kernel 示例',
  noteTitle: '显式表达算子数据流',
  noteCopy: 'RingBuffer 管理片上存储，mem_copy 负责搬运，dsl_matmul 负责计算；Host 侧用 block 数启动 kernel。',
  notePoints: ['GM → L1 → L0A/L0C', '搬运与计算形成流水', '单个入口支持多核调度'],
  fullExample: '查看完整示例',
  tutorial: '入门教程',
  updates: '最近更新',
  releaseTitle: '框架 Release Notes',
  viewAll: '查看全部 →',
  releaseTag: '框架',
  releases: [
    { time: '0.8.3', date: '2026-11-12', title: '排序、归并与矩阵转置', desc: '扩展数据搬运能力，增强运行时诊断。', href: '/zh/release-notes/0.8.3' },
    { time: '0.8.2', date: '2026-11-06', title: 'Tensor 视图与调试能力', desc: '完善搬运、设备函数导出和诊断路径。', href: '/zh/release-notes/0.8.2' },
  ],
  operatorTitle: '算子与网络 Changelog',
  operatorTag: '算子',
  operators: [
    { time: '10.28', date: '2026-10-28', title: '矩阵乘 staging 接口迁移', desc: '统一 RingBuffer、copy_options 与尾块处理。' },
    { time: '09.29', date: '2026-09-29', title: 'attn_prologue', desc: '迁入 MXFP8 attention prologue 与精度用例。' },
  ],
}

const en = {
  subtitle: 'An agent-friendly DSL for Ascend NPU operator development',
  heroCopy: 'Describe data movement, on-chip storage and operator computation in Python, with a programming model, samples and API reference.',
  start: 'Get started',
  repo: 'Source repository',
  startHref: '/en/install/',
  features: [
    { key: 'start', icon: '🔧', title: 'Get started', desc: 'Set up the environment, run an existing sample and write your first operator.', href: '/en/install/' },
    { key: 'guide', icon: '🚀', title: 'Tutorials', desc: 'Learn the language, data and hardware model, then build and engineer operators.', href: '/zh/guide/overview' },
    { key: 'samples', icon: '🧩', title: 'Samples', desc: 'Browse operator source, run instructions and tests by scenario.', href: 'https://gitcode.com/cann/cannbot-dsl/tree/master/samples', external: true },
    { key: 'api', icon: '⚙️', title: 'API reference', desc: 'Host, Kernel and AI CPU interfaces and data structures.', href: '/zh/api/' },
  ],
  quickstart: 'Quick start',
  pip: {
    title: 'Install CANNBot-DSL',
    copy: 'Install the Python package to run the DSL samples and start describing data movement, on-chip storage and computation.',
    start: 'Install',
    quick: 'Quick start →',
    href: '/en/install/',
  },
  workflow: {
    title: 'Install official workflow',
    badge: 'Official',
    copy: 'Add the CANNBot operator workflow to your AI coding tool with the official installer.',
    action: 'View workflow ↗',
  },
  copyText: 'Copy',
  copied: 'Copied',
  copyFailed: 'Copy manually',
  example: 'Core example',
  codeBadge: 'Core snippet',
  codeDoc: 'View docs',
  codeAria: 'Matmul Kernel example',
  noteTitle: 'Explicit operator dataflow',
  noteCopy: 'RingBuffer manages on-chip storage, mem_copy moves data and dsl_matmul computes; the host launches the kernel with a block count.',
  notePoints: ['GM → L1 → L0A/L0C', 'Copy and compute form a pipeline', 'One entry point schedules multiple cores'],
  fullExample: 'Full example',
  tutorial: 'Tutorials',
  updates: 'Recent updates',
  releaseTitle: 'Framework release notes',
  viewAll: 'View all →',
  releaseTag: 'Framework',
  releases: [
    { time: '0.8.3', date: '2026-11-12', title: 'Sorting, merge and matrix transpose', desc: 'Expands data movement and runtime diagnostics.', href: '/zh/release-notes/0.8.3' },
    { time: '0.8.2', date: '2026-11-06', title: 'Tensor views and debugging', desc: 'Refines data movement, device-function export and diagnostics.', href: '/zh/release-notes/0.8.2' },
  ],
  operatorTitle: 'Operator & network changelog',
  operatorTag: 'Operator',
  operators: [
    { time: '10.28', date: '2026-10-28', title: 'Matmul staging API migration', desc: 'Unifies RingBuffer, copy_options and tail-block handling.' },
    { time: '09.29', date: '2026-09-29', title: 'attn_prologue', desc: 'Ports the MXFP8 attention prologue with precision cases.' },
  ],
}

const c = computed(() => (props.lang === 'en' ? en : zh))
const changelogHref = 'https://gitcode.com/cann/cannbot-dsl/blob/master/CHANGELOG.md'

async function copyCommand(id: string, command: string) {
  try {
    await navigator.clipboard.writeText(command)
    copied.value = id
    window.setTimeout(() => {
      if (copied.value === id) copied.value = ''
    }, 1400)
  } catch {
    copied.value = `${id}-failed`
    window.setTimeout(() => {
      if (copied.value === `${id}-failed`) copied.value = ''
    }, 1600)
  }
}
</script>

<template>
  <div class="portal-home">
    <main class="portal-main">
      <section class="portal-hero" aria-labelledby="portal-hero-title">
        <div class="portal-eyebrow">ASCEND NPU · KERNEL DSL</div>
        <h1 id="portal-hero-title">
          <span>CANNBot-DSL</span>
          <span>{{ c.subtitle }}</span>
        </h1>
        <p class="portal-hero-copy">{{ c.heroCopy }}</p>
        <div class="portal-actions">
          <a class="portal-button primary" :href="c.startHref">{{ c.start }}</a>
          <a class="portal-button" href="https://gitcode.com/cann/cannbot-dsl" target="_blank" rel="noopener">{{ c.repo }}</a>
        </div>
      </section>

      <section class="portal-features" aria-label="Features">
        <a
          v-for="f in c.features"
          :key="f.key"
          class="portal-feature"
          :href="f.href"
          :target="f.external ? '_blank' : undefined"
          :rel="f.external ? 'noopener' : undefined"
        >
          <span class="portal-feature-icon" aria-hidden="true">{{ f.icon }}</span>
          <h2>{{ f.title }}</h2>
          <p>{{ f.desc }}</p>
        </a>
      </section>

      <section class="portal-section portal-quickstart" aria-labelledby="portal-quickstart-title">
        <div class="portal-section-head">
          <h2 id="portal-quickstart-title">{{ c.quickstart }}</h2>
        </div>
        <div class="portal-quick-grid">
          <article class="portal-card portal-quick-card">
            <div class="portal-card-head">
              <div class="portal-card-title">
                <span class="portal-command-mark">PY</span>
                <div>
                  <h3>{{ c.pip.title }}</h3>
                </div>
              </div>
              <span class="portal-badge">pip</span>
            </div>
            <p class="portal-card-copy">{{ c.pip.copy }}</p>
            <div class="portal-command-wrap">
              <code><span class="portal-prompt">$ </span>{{ pipCommand }}</code>
              <button class="portal-copy" type="button" @click="copyCommand('pip', pipCommand)">
                {{ copied === 'pip' ? c.copied : copied === 'pip-failed' ? c.copyFailed : c.copyText }}
              </button>
            </div>
            <div class="portal-card-actions">
              <a class="portal-button primary small" :href="c.pip.href">{{ c.pip.start }}</a>
              <a class="portal-inline-link" :href="c.pip.href">{{ c.pip.quick }}</a>
            </div>
          </article>

          <article class="portal-card portal-quick-card portal-workflow-card">
            <div class="portal-card-head">
              <div class="portal-card-title">
                <span class="portal-command-mark">CLI</span>
                <div>
                  <h3>{{ c.workflow.title }}</h3>
                </div>
              </div>
              <span class="portal-badge portal-badge-brand">{{ c.workflow.badge }}</span>
            </div>
            <p class="portal-card-copy">{{ c.workflow.copy }}</p>
            <div class="portal-command-wrap">
              <code><span class="portal-prompt">$ </span>{{ workflowCommand }}</code>
              <button class="portal-copy" type="button" @click="copyCommand('workflow', workflowCommand)">
                {{ copied === 'workflow' ? c.copied : copied === 'workflow-failed' ? c.copyFailed : c.copyText }}
              </button>
            </div>
            <div class="portal-card-actions">
              <a class="portal-button primary small" href="https://gitcode.com/cann/cannbot" target="_blank" rel="noopener">{{ c.workflow.action }}</a>
              <a class="portal-inline-link" href="https://gitcode.com/cann/cannbot" target="_blank" rel="noopener">cann/cannbot</a>
            </div>
          </article>
        </div>
      </section>

      <section class="portal-section" aria-labelledby="portal-example-title">
        <div class="portal-section-head">
          <h2 id="portal-example-title">{{ c.example }}</h2>
        </div>
        <article class="portal-card portal-code-card">
          <div class="portal-code-head">
            <div class="portal-code-title">
              <span class="portal-terminal-dots" aria-hidden="true"><i></i><i></i><i></i></span>
              <span>Matmul Kernel</span>
              <small>{{ c.codeBadge }}</small>
            </div>
            <div class="portal-code-head-right">
              <span class="portal-code-meta">Kernel + Host · Ascend 950</span>
              <a class="portal-code-doc" href="/zh/guide/first-operator">{{ c.codeDoc }}</a>
            </div>
          </div>
          <div class="portal-code-body">
            <div class="portal-code-note">
              <h3>{{ c.noteTitle }}</h3>
              <p>{{ c.noteCopy }}</p>
              <ul>
                <li v-for="point in c.notePoints" :key="point">{{ point }}</li>
              </ul>
              <div class="portal-card-actions">
                <a class="portal-button primary small" href="/zh/guide/first-operator">{{ c.fullExample }}</a>
                <a class="portal-button small" href="/zh/guide/overview">{{ c.tutorial }}</a>
              </div>
            </div>
            <div class="portal-code-wrap">
              <pre :aria-label="c.codeAria"><code><span class="portal-code-decorator">@kernel</span>
<span class="portal-code-keyword">def</span> <span class="portal-code-function">matmul_kernel</span>(gm_a, gm_b, gm_c):
    dt = gm_a.dtype
    l1a = <span class="portal-code-type">RingBuffer</span>(<span class="portal-code-type">MemLoc</span>.<span class="portal-code-property">L1</span>, dt, <span class="portal-code-function">make_layout</span>((bm, kl)), <span class="portal-code-property">depth</span>=<span class="portal-code-number">2</span>)
    l1b = <span class="portal-code-type">RingBuffer</span>(<span class="portal-code-type">MemLoc</span>.<span class="portal-code-property">L1</span>, dt, <span class="portal-code-function">make_layout</span>((bn, kl)), <span class="portal-code-property">depth</span>=<span class="portal-code-number">2</span>)
    l0a = <span class="portal-code-type">RingBuffer</span>(<span class="portal-code-type">MemLoc</span>.<span class="portal-code-property">L0A</span>, dt, <span class="portal-code-function">make_layout</span>((bm, bk)), <span class="portal-code-property">depth</span>=<span class="portal-code-number">2</span>)
    l0b = <span class="portal-code-type">RingBuffer</span>(<span class="portal-code-type">MemLoc</span>.<span class="portal-code-property">L0B</span>, dt, <span class="portal-code-function">make_layout</span>((bn, bk)), <span class="portal-code-property">depth</span>=<span class="portal-code-number">2</span>)
    l0c = <span class="portal-code-type">RingBuffer</span>(<span class="portal-code-type">MemLoc</span>.<span class="portal-code-property">L0C</span>, <span class="portal-code-type">dtypes</span>.<span class="portal-code-property">float32</span>, <span class="portal-code-function">make_layout</span>((bm, bn)))
    <span class="portal-code-keyword">for</span> t <span class="portal-code-keyword">in</span> <span class="portal-code-function">range</span>(<span class="portal-code-function">get_block_idx</span>(), tiles, <span class="portal-code-function">get_block_num</span>()):
        <span class="portal-code-function">mem_copy</span>(l1a.<span class="portal-code-function">next_write</span>(), <span class="portal-code-function">tile_slice</span>(gm_a, (bm, kl), (t, <span class="portal-code-number">0</span>)))
        <span class="portal-code-function">mem_copy</span>(l1b.<span class="portal-code-function">next_write</span>(), <span class="portal-code-function">tile_slice</span>(gm_b, (bn, kl), (t, <span class="portal-code-number">0</span>)))
        <span class="portal-code-function">mem_copy</span>(l0a.<span class="portal-code-function">next_write</span>(), l1a.<span class="portal-code-function">next_read</span>())
        <span class="portal-code-function">mem_copy</span>(l0b.<span class="portal-code-function">next_write</span>(), l1b.<span class="portal-code-function">next_read</span>())
        acc = <span class="portal-code-function">dsl_matmul</span>(l0c.<span class="portal-code-function">next_write</span>(), l0a.<span class="portal-code-function">next_read</span>(), l0b.<span class="portal-code-function">next_read</span>())
        <span class="portal-code-function">mem_copy</span>(<span class="portal-code-function">tile_slice</span>(gm_c, (bm, bn), (t, <span class="portal-code-number">0</span>)), acc)

<span class="portal-code-decorator">@host</span>
<span class="portal-code-keyword">def</span> <span class="portal-code-function">run</span>(gm_a, gm_b, gm_c, block_dim):
    <span class="portal-code-function">matmul_kernel</span>[block_dim](gm_a, gm_b, gm_c)</code></pre>
            </div>
          </div>
        </article>
      </section>

      <section class="portal-section portal-updates" aria-labelledby="portal-updates-title">
        <div class="portal-section-head">
          <h2 id="portal-updates-title">{{ c.updates }}</h2>
        </div>
        <div class="portal-update-grid">
          <article class="portal-update-list">
            <div class="portal-update-head">
              <h3>{{ c.releaseTitle }}</h3>
              <a href="/zh/release-notes">{{ c.viewAll }}</a>
            </div>
            <a v-for="r in c.releases" :key="r.time" class="portal-update-row" :href="r.href">
              <time :datetime="r.date">{{ r.time }}</time>
              <span><strong>{{ r.title }}</strong><small>{{ r.desc }}</small></span>
              <em>{{ c.releaseTag }}</em>
            </a>
          </article>
          <article class="portal-update-list portal-update-operator">
            <div class="portal-update-head">
              <h3>{{ c.operatorTitle }}</h3>
              <a :href="changelogHref" target="_blank" rel="noopener">{{ c.viewAll }}</a>
            </div>
            <a v-for="o in c.operators" :key="o.time" class="portal-update-row" :href="changelogHref" target="_blank" rel="noopener">
              <time :datetime="o.date">{{ o.time }}</time>
              <span><strong>{{ o.title }}</strong><small>{{ o.desc }}</small></span>
              <em>{{ c.operatorTag }}</em>
            </a>
          </article>
        </div>
      </section>
    </main>
  </div>
</template>
