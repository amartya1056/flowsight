/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: { 950: '#07090d', 900: '#0b0f15', 850: '#0f141c', 800: '#131a24', 700: '#1b2430', 600: '#263142', 500: '#3a475a', 400: '#5d6b80', 300: '#8b97a8', 200: '#b9c2cf', 100: '#e3e8ef' },
        signal: { DEFAULT: '#2dd4bf', dim: '#14796d' },
        sev: { critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#64748b' },
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
