function align2seq(seq1, seq2) {
  const matchScore = 1;
  const mismatchScore = -1;
  const gapPenalty = -1;

  const m = seq1.length;
  const n = seq2.length;

  const score = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  const traceback = Array.from({ length: m + 1 }, () => Array(n + 1).fill(""));

  // Initialize borders
  for (let i = 0; i <= m; i++) {
    score[i][0] = i * gapPenalty;
    traceback[i][0] = "U"; // up
  }
  for (let j = 0; j <= n; j++) {
    score[0][j] = j * gapPenalty;
    traceback[0][j] = "L"; // left
  }
  traceback[0][0] = "E"; // end/start

  // Fill matrices
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const matchMismatch = score[i - 1][j - 1] + (seq1[i - 1] === seq2[j - 1] ? matchScore : mismatchScore);
      const deleteOp = score[i - 1][j] + gapPenalty;
      const insertOp = score[i][j - 1] + gapPenalty;

      const maxScore = Math.max(matchMismatch, deleteOp, insertOp);
      score[i][j] = maxScore;

      if (maxScore === matchMismatch) traceback[i][j] = "D"; // diagonal
      else if (maxScore === deleteOp) traceback[i][j] = "U"; // up
      else traceback[i][j] = "L"; // left
    }
  }

  // Traceback
  let alignedSeq1 = "";
  let alignedSeq2 = "";
  let matchLine = "";

  let i = m, j = n;
  while (!(i === 0 && j === 0)) {
    if (traceback[i][j] === "D") {
      alignedSeq1 = seq1[i - 1] + alignedSeq1;
      alignedSeq2 = seq2[j - 1] + alignedSeq2;
      matchLine = (seq1[i - 1] === seq2[j - 1] ? "|" : ":") + matchLine;
      i--;
      j--;
    } else if (traceback[i][j] === "U") {
      alignedSeq1 = seq1[i - 1] + alignedSeq1;
      alignedSeq2 = "-" + alignedSeq2;
      matchLine = "." + matchLine;
      i--;
    } else if (traceback[i][j] === "L") {
      alignedSeq1 = "-" + alignedSeq1;
      alignedSeq2 = seq2[j - 1] + alignedSeq2;
      matchLine = "." + matchLine;
      j--;
    }
  }

  return {
    alignedSeq1,
    alignedSeq2,
    matchLine,
    score: score[m][n],
  };
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