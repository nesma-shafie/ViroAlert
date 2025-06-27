"use client";
import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Dna, Info, Search } from "lucide-react";
import {
  Tooltip,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { TooltipContent } from "@radix-ui/react-tooltip";

export default function SequenceMatchViewer() {
  const sequenceData = JSON.parse(localStorage.getItem("sequenceData") || "{}");
  const [selectedMatch, setSelectedMatch] = useState(
    sequenceData.closest_matches[0]
  );
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const getScoreColor = (jaccard: number) => {
    if (jaccard >= 0.85) return "from-green-500 to-emerald-600";
    if (jaccard >= 0.8) return "from-blue-500 to-cyan-600";
    if (jaccard >= 0.75) return "from-yellow-500 to-orange-600";
    return "from-red-500 to-pink-600";
  };

  const getScoreBadgeColor = (jaccard: number) => {
    if (jaccard >= 0.85) return "bg-green-100 text-green-800 border-green-200";
    if (jaccard >= 0.8) return "bg-blue-100 text-blue-800 border-blue-200";
    if (jaccard >= 0.75)
      return "bg-yellow-100 text-yellow-800 border-yellow-200";
    return "bg-red-100 text-red-800 border-red-200";
  };

  return (
    <TooltipProvider>
      <div className="min-h-screen bg-gradient-to-br from-gray-100 via-blue-100 to-slate-200 p-6">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Enhanced Header */}
          <div className="text-center space-y-4 relative">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-400/20 to-blue-600/20 blur-3xl -z-10"></div>
            <div className="flex items-center justify-center space-x-3">
              <Dna className="w-10 h-10 text-cyan-600 animate-pulse" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
                Sequence Match Analyzer
              </h1>
              <Search className="w-8 h-8 text-blue-600 animate-bounce" />
            </div>
            <p className="text-gray-600 text-lg">
              Advanced viral sequence alignment and similarity analysis
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Input Sequence Panel */}
            <div className="lg:col-span-1 space-y-6">
              <Card className="border-0 shadow-xl bg-gradient-to-br from-white to-gray-50 hover:shadow-2xl transition-all duration-300">
                <CardContent className="p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-3 h-3 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full animate-pulse"></div>
                    <h2 className="text-xl font-bold text-gray-800">
                      Input Sequence
                    </h2>
                  </div>
                  <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg p-4 shadow-inner">
                    <ScrollArea className="h-40">
                      <div className="font-mono text-xs text-green-400 leading-relaxed break-all">
                        {sequenceData.input
                          .split("")
                          .map((char: string, i: number) => (
                            <span
                              key={i}
                              className="hover:bg-green-500/20 hover:text-green-300 transition-colors duration-150 cursor-default"
                            >
                              {char}
                            </span>
                          ))}
                      </div>
                    </ScrollArea>
                  </div>
                  <div className="mt-4 flex justify-between text-sm text-gray-600">
                    <span>Length: {sequenceData.input.length} </span>
                    <span>Type: Protein</span>
                  </div>
                </CardContent>
              </Card>

              {/* Match Statistics */}
              <Card className="border-0 shadow-xl bg-gradient-to-br from-white to-blue-50">
                <CardContent className="p-6">
                  <h3 className="text-lg font-bold text-gray-800 mb-4">
                    Match Statistics
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Best Match</span>

                        <span className="font-semibold">
                          {(selectedMatch.jaccard * 100).toFixed(1)}%
                        </span>
                      </div>
                      <Progress
                        value={selectedMatch.jaccard * 100}
                        className="h-2 bg-gray-200"
                      />
                    </div>
                    <div className="grid grid-cols-1 gap-4 text-center">
                      <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-3">
                        <div className="text-2xl font-bold text-green-600">
                          {selectedMatch.score}
                        </div>

                        <div className="text-xs text-gray-600">
                          Alignment Score
                        </div>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Info className="h-4 w-4 text-gray-500 cursor-help" />
                          </TooltipTrigger>
                          <TooltipContent>
                            <p className="text-white-sm max-w-xs">
                              The alignment score indicates the overall quality of the match, with higher values indicating better alignment.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Matches List */}
            <div className="lg:col-span-2 space-y-6">
              <Card className="border-0 shadow-xl bg-gradient-to-br from-white to-indigo-50">
                <CardContent className="p-6">
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="w-3 h-3 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full animate-pulse"></div>
                    <h2 className="text-xl font-bold text-gray-800">
                      Top Sequence Matches
                    </h2>
                  </div>

                  <div className="space-y-3 max-h-80 overflow-y-auto">
                    {sequenceData.closest_matches.map(
                      (match: any, idx: any) => (
                        <div
                          key={idx}
                          onClick={() => setSelectedMatch(match)}
                          onMouseEnter={() => setHoveredIndex(idx)}
                          onMouseLeave={() => setHoveredIndex(null)}
                          className={`relative p-4 rounded-xl border-2 cursor-pointer transition-all duration-300 ${match === selectedMatch
                              ? "border-blue-500 bg-gradient-to-r from-blue-50 to-indigo-50 shadow-lg transform scale-[1.02]"
                              : "border-gray-200 bg-white hover:border-blue-300 hover:shadow-md hover:transform hover:scale-[1.01]"
                            }`}
                        >
                          <div className="flex items-center justify-between mb-3">
                            <Badge
                              variant="outline"
                              className={`${getScoreBadgeColor(
                                match.jaccard
                              )} font-semibold px-3 py-1`}
                            >
                              Match #{idx + 1}
                            </Badge>
                            <div className="flex items-center space-x-2">
                              <div
                                className={`text-xs px-2 py-1 rounded-full bg-gradient-to-r ${getScoreColor(
                                  match.jaccard
                                )} text-white font-bold`}
                              >
                                {(match.jaccard * 100).toFixed(1)}%
                              </div>
                              <div className="text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-700 font-semibold">
                                Score: {match.score}
                              </div>
                            </div>
                          </div>

                          <div className="font-mono text-sm text-gray-700 leading-relaxed">
                            {match.label}
                          </div>

                          <div className="mt-3">
                            <div className="flex justify-between text-xs text-gray-500 mb-1">
                              <span>Similarity</span>
                              <span>{(match.jaccard * 100).toFixed(2)}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                              <div
                                className={`h-2 bg-gradient-to-r ${getScoreColor(
                                  match.jaccard
                                )} transition-all duration-500 ease-out`}
                                style={{
                                  width: `${match.jaccard * 100}%`,
                                  transform:
                                    hoveredIndex === idx
                                      ? "scaleY(1.2)"
                                      : "scaleY(1)",
                                }}
                              ></div>
                            </div>
                          </div>

                          {match === selectedMatch && (
                            <div className="absolute -left-1 top-0 bottom-0 w-1 bg-gradient-to-b from-blue-500 to-indigo-600 rounded-full animate-pulse"></div>
                          )}
                        </div>
                      )
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Enhanced Alignment Viewer */}
              <Card className="border-0 shadow-xl bg-gradient-to-br from-white to-purple-50">
                <CardContent className="p-6">
                  <div className="flex items-center space-x-3 mb-6">
                    <div className="w-3 h-3 bg-gradient-to-r from-purple-500 to-pink-600 rounded-full animate-pulse"></div>
                    <h2 className="text-xl font-bold text-gray-800">
                      Sequence Alignment
                    </h2>
                  </div>

                  <div className="bg-gradient-to-r from-gray-900 to-black rounded-xl p-6 shadow-inner">
                    <div className="mb-4 text-xs text-gray-400 font-mono">
                      Selected: {selectedMatch.label}
                    </div>
                    <EnhancedAlignmentView
                      seq1={selectedMatch.alignedSeq1}
                      seq2={selectedMatch.alignedSeq2}
                      matchline={selectedMatch.matchLine}
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
// Enhanced alignment viewer with color coding and better visualization
function EnhancedAlignmentView({
  seq1,
  seq2,
  matchline
}: {
  seq1: string;
  seq2: string;
  matchline: string;
}) {
  const chunkSize = 60;
  const chunks = [];

  for (let i = 0; i < seq1.length; i += chunkSize) {
    chunks.push({
      seq1: seq1.slice(i, i + chunkSize),
      seq2: seq2.slice(i, i + chunkSize),
      matchline: matchline.slice(i, i + chunkSize),
      position: i
    });
  }

  const getCharColor = (char: string, isMatch: boolean, isSeq1: boolean) => {
    if (char === '-') {
      return 'text-red-500 bg-gray-800/50';
    }
    if (isMatch) {
      return isSeq1 ? 'text-green-400 bg-green-900/30' : 'text-cyan-400 bg-cyan-900/30';
    }
    return isSeq1 ? 'text-red-400 bg-red-900/20' : 'text-orange-400 bg-orange-900/20';
  };

  const getMatchlineChar = (char: string) => {
    if (char === '|') return 'text-green-400';
    if (char === ':') return 'text-yellow-400';
    if (char === '.') return 'text-blue-400';
    return 'text-gray-600';
  };

  return (
    <ScrollArea className="h-96">
      <div className="space-y-6 font-mono text-xs">
        {chunks.map((chunk, chunkIndex) => (
          <div key={chunkIndex} className="space-y-1">
            {/* Position indicators */}
            <div className="text-gray-500 text-[10px] mb-1">
              Position: {chunk.position + 1} - {chunk.position + chunk.seq1.length}
            </div>

            {/* Query sequence */}
            <div className="flex items-center space-x-2">
              <span className="text-gray-400 w-12 text-[10px]">Query:</span>
              <div className="flex">
                {chunk.seq1.split('').map((char, i) => {
                  const isMatch = chunk.matchline[i] === '|';
                  return (
                    <span
                      key={i}
                      className={`px-[1px] py-[1px] rounded transition-all duration-200 hover:scale-110 ${getCharColor(char, isMatch, true)}`}
                      title={`Position: ${chunk.position + i + 1}, Residue: ${char}`}
                    >
                      {char}
                    </span>
                  );
                })}
              </div>
            </div>

            {/* Match line */}
            <div className="flex items-center space-x-2">
              <span className="text-gray-400 w-12 text-[10px]">Match:</span>
              <div className="flex">
                {chunk.matchline.split('').map((char, i) => (
                  <span
                    key={i}
                    className={`px-[1px] py-[1px] ${getMatchlineChar(char)}`}
                    title={
                      char === '|' ? 'Exact match' :
                        char === ':' ? 'changed residues' :
                          char === '.' ? 'skipped residues' :
                            'No similarity'
                    }
                  >
                    {char}
                  </span>
                ))}
              </div>
            </div>

            {/* Subject sequence */}
            <div className="flex items-center space-x-2">
              <span className="text-gray-400 w-12 text-[10px]">Subj:</span>
              <div className="flex">
                {chunk.seq2.split('').map((char, i) => {
                  const isMatch = chunk.matchline[i] === '|';
                  return (
                    <span
                      key={i}
                      className={`px-[1px] py-[1px] rounded transition-all duration-200 hover:scale-110 ${getCharColor(char, isMatch, false)}`}
                      title={`Position: ${chunk.position + i + 1}, Residue: ${char}`}
                    >
                      {char}
                    </span>
                  );
                })}
              </div>
            </div>

            {/* Statistics for this chunk */}
            <div className="text-[10px] text-gray-500 pt-1 border-t border-gray-700/50">
              Matches: {chunk.matchline.split('').filter(c => c === '|').length}/{chunk.matchline.length}
              ({((chunk.matchline.split('').filter(c => c === '|').length / chunk.matchline.length) * 100).toFixed(1)}%)
            </div>
          </div>
        ))}
      </div>
    </ScrollArea>
  );
}
