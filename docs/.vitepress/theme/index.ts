import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import { useData } from 'vitepress'
import './custom.css'

const PREVIEW_NOTICE =
  '当前为尝鲜版本，CANNBot-DSL 的 API 接口不保证兼容性，后续版本可能发生变更。'

export default {
  extends: DefaultTheme,
  Layout() {
    const { page } = useData()

    return h(DefaultTheme.Layout, null, {
      'doc-before': () => {
        if (!page.value.relativePath.startsWith('zh/api/')) return null
        return h('p', { class: 'preview-notice' }, PREVIEW_NOTICE)
      },
    })
  },
}
