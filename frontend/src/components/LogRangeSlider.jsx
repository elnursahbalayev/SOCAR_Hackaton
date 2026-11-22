// Two-handle slider on a LOG scale (time since shut-in spans minutes to days, so a log scale is natural).
const STEPS = 1000

function toPos(value, min, max) {
  return Math.round(((Math.log10(value) - Math.log10(min)) / (Math.log10(max) - Math.log10(min))) * STEPS)
}

function fromPos(pos, min, max) {
  return 10 ** (Math.log10(min) + (pos / STEPS) * (Math.log10(max) - Math.log10(min)))
}

export default function LogRangeSlider({ min, max, start, end, onChange }) {
  const a = toPos(start, min, max)
  const b = toPos(end, min, max)
  const thumb = `pointer-events-none absolute inset-0 h-1.5 w-full appearance-none bg-transparent
    [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4
    [&::-webkit-slider-thumb]:cursor-grab [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full
    [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-bg [&::-webkit-slider-thumb]:bg-accent
    [&::-webkit-slider-thumb]:shadow-[0_0_0_4px_rgba(245,181,68,0.18)]
    [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4
    [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:bg-accent`
  return (
    <div className="relative h-4">
      <div className="absolute top-1/2 h-1.5 w-full -translate-y-1/2 rounded-full bg-white/8" />
      <div className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-accent/70"
        style={{ left: `${(a / STEPS) * 100}%`, width: `${((b - a) / STEPS) * 100}%` }} />
      <input type="range" min={0} max={STEPS} value={a} className={`${thumb} top-1/2 -translate-y-1/2`}
        onChange={(e) => onChange(fromPos(Math.min(+e.target.value, b - 20), min, max), end)} />
      <input type="range" min={0} max={STEPS} value={b} className={`${thumb} top-1/2 -translate-y-1/2`}
        onChange={(e) => onChange(start, fromPos(Math.max(+e.target.value, a + 20), min, max))} />
    </div>
  )
}
