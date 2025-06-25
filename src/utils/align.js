function align2seq(seq1, seq2) {
  const matchScore = 1;
  const mismatchScore = -1;
  const gapPenalty = -1;

  const m = seq1.length;
  const n = seq2.length;

  let prevRow = new Array(n + 1);
  let currRow = new Array(n + 1);

  // Initialize first row
  for (let j = 0; j <= n; j++) {
    prevRow[j] = j * gapPenalty;
  }

  for (let i = 1; i <= m; i++) {
    currRow[0] = i * gapPenalty;
    for (let j = 1; j <= n; j++) {
      const match = prevRow[j - 1] + (seq1[i - 1] === seq2[j - 1] ? matchScore : mismatchScore);
      const deleteOp = prevRow[j] + gapPenalty;
      const insertOp = currRow[j - 1] + gapPenalty;
      currRow[j] = Math.max(match, deleteOp, insertOp);
    }
    // Swap rows
    [prevRow, currRow] = [currRow, prevRow];
  }

  return prevRow[n]; // Final alignment score
}

function getKmers(sequence, k = 5) {
  const kmers = new Set();
  for (let i = 0; i <= sequence.length - k; i++) {
    kmers.add(sequence.slice(i, i + k));
  }
  return kmers;
}
function jaccardSimilarity(setA, setB) {
  const intersection = new Set([...setA].filter(x => setB.has(x)));
  const union = new Set([...setA, ...setB]);
  return intersection.size / union.size;
}

export { align2seq, getKmers, jaccardSimilarity };