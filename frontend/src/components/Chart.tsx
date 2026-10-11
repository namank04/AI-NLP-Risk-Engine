import type { EChartsOption } from 'echarts'
import ReactECharts from 'echarts-for-react'
import { AXIS, GRID, INK, SURFACE } from '../lib/theme'

// Shared chart chrome: hairline solid grid, muted axis text, quiet tooltip.
// Marks carry colour; text never does.
const axis = {
  axisLine: { lineStyle: { color: AXIS } },
  axisTick: { show: false },
  axisLabel: { color: INK.muted, fontSize: 11 },
  splitLine: { lineStyle: { color: GRID, width: 1 } },
}

// static captures (?snapshot) render without animation so screenshots show final marks
const STATIC = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('snapshot')

export function themed(o: EChartsOption): EChartsOption {
  const merge = (a: unknown) => (Array.isArray(a) ? a.map((x) => ({ ...axis, ...x })) : a ? { ...axis, ...(a as object) } : undefined)
  return {
    backgroundColor: 'transparent',
    animation: !STATIC,
    animationDuration: 700,
    animationEasing: 'cubicOut',
    textStyle: { fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif', color: INK.secondary },
    grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true },
    ...o,
    tooltip: {
      backgroundColor: SURFACE.raised,
      borderColor: '#2b3240',
      borderWidth: 1,
      padding: [8, 10],
      textStyle: { color: INK.primary, fontSize: 12 },
      extraCssText: 'border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.35);',
      ...(o.tooltip as object),
    },
    legend: o.legend ? { textStyle: { color: INK.secondary, fontSize: 12 }, icon: 'roundRect', itemWidth: 10, itemHeight: 10, ...(o.legend as object) } : undefined,
    xAxis: merge(o.xAxis) as EChartsOption['xAxis'],
    yAxis: merge(o.yAxis) as EChartsOption['yAxis'],
  }
}

export function Chart({ option, height = 260, onClick, merge }: { option: EChartsOption; height?: number; onClick?: (p: unknown) => void; merge?: boolean }) {
  return (
    <ReactECharts option={themed(option)} style={{ height, width: '100%' }} notMerge={!merge} lazyUpdate
      onEvents={onClick ? { click: onClick } : undefined} opts={{ renderer: 'canvas' }} />
  )
}

/** Bar mark spec: <= 24px thick, 4px rounded data-end, square at the baseline. */
export const barStyle = (horizontal: boolean, positive = true) => ({
  barMaxWidth: 22,
  itemStyle: {
    borderRadius: horizontal ? (positive ? [0, 4, 4, 0] : [4, 0, 0, 4]) : positive ? [4, 4, 0, 0] : [0, 0, 4, 4],
  },
})
