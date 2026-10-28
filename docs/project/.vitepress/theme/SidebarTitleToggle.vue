<script setup lang="ts">
/*
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
  */

import { nextTick, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vitepress'

const router = useRouter()
const selector = '.VPSidebarItem.collapsible > .item > a.link'
let pending: { href: string; shouldOpen: boolean; keepSidebarOpen: boolean } | null = null
const previousAfterRouteChanged = router.onAfterRouteChanged

async function applyToggle() {
  const request = pending
  if (!request) return
  pending = null
  await nextTick()
  const currentLink = Array.from(document.querySelectorAll<HTMLAnchorElement>(selector))
    .find(item => item.href === request.href)
  const currentGroup = currentLink?.parentElement?.parentElement
  if (currentGroup && currentGroup.classList.contains('collapsed') === request.shouldOpen) {
    currentLink?.parentElement?.querySelector<HTMLElement>(':scope > .caret')?.click()
  }
  if (request.keepSidebarOpen && !document.querySelector('.VPSidebar.open')) {
    document.querySelector<HTMLButtonElement>('button[aria-controls="VPSidebarNav"]')?.click()
  }
}

const afterRouteChanged: NonNullable<typeof router.onAfterRouteChanged> = async to => {
  await previousAfterRouteChanged?.(to)
  await applyToggle()
}

function onTitleClick(event: MouseEvent) {
  // VitePress's window capture handler already prevents the link's default
  // action. Observe that navigation rather than issuing a second one.
  if (event.button !== 0 || event.ctrlKey ||
      event.metaKey || event.shiftKey || event.altKey) return
  if (!(event.target instanceof Element)) return
  const link = event.target.closest<HTMLAnchorElement>(selector)
  if (!link) return
  const url = new URL(link.href)
  if (url.origin !== location.origin ||
      !/^\/zh\/api\/kernel\/types-and-views\/[^/]+\/(?:index(?:\.html)?)?$/.test(url.pathname)) return

  const group = link.parentElement?.parentElement
  if (!group) return
  pending = {
    href: link.href,
    shouldOpen: group.classList.contains('collapsed'),
    keepSidebarOpen: !!document.querySelector('.VPSidebar.open')
  }
  if (url.pathname === location.pathname && url.search === location.search) {
    void applyToggle()
  } else if (!event.defaultPrevented) {
    event.preventDefault()
    void router.go(link.href)
  }
}

onMounted(() => {
  router.onAfterRouteChanged = afterRouteChanged
  document.addEventListener('click', onTitleClick, true)
})
onUnmounted(() => {
  document.removeEventListener('click', onTitleClick, true)
  if (router.onAfterRouteChanged === afterRouteChanged) {
    router.onAfterRouteChanged = previousAfterRouteChanged
  }
})
</script>

<template><span hidden aria-hidden="true" /></template>
