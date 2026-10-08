// Tiny stroke font for neon signs. Glyphs live on a 4 x 6 grid (y up); each
// glyph is a list of polylines.

type Glyph = number[][];

const STAR: number[] = (() => {
  const pts: number[] = [];
  for (let i = 0; i <= 10; i++) {
    const a = Math.PI / 2 + (i * Math.PI) / 5;
    const r = i % 2 === 0 ? 3 : 1.25;
    pts.push(2 + Math.cos(a) * r, 3 + Math.sin(a) * r);
  }
  return pts;
})();

export const GLYPHS: Record<string, Glyph> = {
  A: [[0, 0, 2, 6, 4, 0], [0.9, 2.4, 3.1, 2.4]],
  B: [[0, 0, 0, 6, 3, 6, 4, 5, 4, 4, 3, 3, 0, 3], [3, 3, 4, 2, 4, 1, 3, 0, 0, 0]],
  C: [[4, 5, 3, 6, 1, 6, 0, 5, 0, 1, 1, 0, 3, 0, 4, 1]],
  E: [[4, 6, 0, 6, 0, 0, 4, 0], [0, 3, 3, 3]],
  G: [[4, 5, 3, 6, 1, 6, 0, 5, 0, 1, 1, 0, 3, 0, 4, 1, 4, 3, 2, 3]],
  I: [[2, 0, 2, 6], [1, 0, 3, 0], [1, 6, 3, 6]],
  J: [[1, 6, 4, 6], [3, 6, 3, 1, 2, 0, 1, 0, 0, 1]],
  K: [[0, 0, 0, 6], [4, 6, 0, 2.5], [1.2, 3.4, 4, 0]],
  L: [[0, 6, 0, 0, 4, 0]],
  N: [[0, 0, 0, 6, 4, 0, 4, 6]],
  O: [[1, 0, 0, 1, 0, 5, 1, 6, 3, 6, 4, 5, 4, 1, 3, 0, 1, 0]],
  P: [[0, 0, 0, 6, 3, 6, 4, 5, 4, 4, 3, 3, 0, 3]],
  R: [[0, 0, 0, 6, 3, 6, 4, 5, 4, 4, 3, 3, 0, 3], [2, 3, 4, 0]],
  S: [[4, 5, 3, 6, 1, 6, 0, 5, 0, 4, 1, 3, 3, 3, 4, 2, 4, 1, 3, 0, 1, 0, 0, 1]],
  T: [[0, 6, 4, 6], [2, 6, 2, 0]],
  U: [[0, 6, 0, 1, 1, 0, 3, 0, 4, 1, 4, 6]],
  V: [[0, 6, 2, 0, 4, 6]],
  X: [[0, 0, 4, 6], [0, 6, 4, 0]],
  Y: [[0, 6, 2, 3, 4, 6], [2, 3, 2, 0]],
  '7': [[0, 6, 4, 6, 1.5, 0]],
  '*': [STAR],
  ' ': [],
};

export interface Stroke {
  a: [number, number];
  b: [number, number];
}

/** Lay out text as 2D segments in glyph units (height 6, advance 5.5). */
export function layoutText(text: string): { strokes: Stroke[]; width: number } {
  const strokes: Stroke[] = [];
  let x = 0;
  for (const ch of text.toUpperCase()) {
    const g = GLYPHS[ch] ?? [];
    for (const line of g) {
      for (let i = 0; i + 3 < line.length; i += 2) {
        strokes.push({ a: [x + line[i], line[i + 1]], b: [x + line[i + 2], line[i + 3]] });
      }
    }
    x += 5.5;
  }
  return { strokes, width: Math.max(0, x - 1.5) };
}
