"use client";

import { useEffect, useState } from "react";
import Typewriter from "typewriter-effect";
import { Card, CardContent } from "@/components/ui/card";
import { BrainCog, ShieldCheck, FlaskConical } from "lucide-react";

const iconProps = { className: "w-6 h-6 text-slate-700" };

interface Particle {
  id: number;
  left: string;
  top: string;
  duration: string;
  delay: string;
}

export default function AboutPage() {
  const [particles, setParticles] = useState<Particle[]>([]);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    // Generate particles only on client side to avoid hydration mismatch
    const generatedParticles = [...Array(20)].map((_, i) => ({
      id: i,
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      duration: `${3 + Math.random() * 4}s`,
      delay: `${Math.random() * 2}s`,
    }));
    setParticles(generatedParticles);
  }, []);

  const features = [
    {
      icon: BrainCog,
      title: "AI-Driven Prediction",
      description:
        "Utilizing advanced machine learning and deep learning techniques to predict the pandemic potential of emerging viruses with high accuracy.",
    },
    {
      icon: ShieldCheck,
      title: "Drug Generation",
      description:
        "Employing structural embeddings and domain-specific fine-tuning to generate novel antiviral candidates that effectively target viral proteins.",
    },
    {
      icon: FlaskConical,
      title: "Biological Relevance",
      description:
        "Utilizes Drug-Target Interaction (DTI) models to rank generated molecules by biological efficacy and identify top-performing commercial antivirals.",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 via-blue-100 to-slate-200 relative overflow-hidden">
      {/* Enhanced Animated Background Grid */}
      <div className="absolute inset-0 opacity-20">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `
            linear-gradient(rgba(59, 130, 246, 0.2) 1px, transparent 1px),
            linear-gradient(90deg, rgba(59, 130, 246, 0.2) 1px, transparent 1px)
          `,
            backgroundSize: "50px 50px",
            animation: "grid-move 20s linear infinite",
          }}
        ></div>
      </div>

      {/* Floating Particles - Only render on client */}
      {isClient && (
        <div className="absolute inset-0 overflow-hidden">
          {particles.map((particle) => (
            <div
              key={particle.id}
              className="absolute w-1 h-1 bg-blue-500 rounded-full opacity-40"
              style={{
                left: particle.left,
                top: particle.top,
                animation: `float ${particle.duration} ease-in-out infinite`,
                animationDelay: particle.delay,
              }}
            ></div>
          ))}
        </div>
      )}

      {/* Enhanced Scientific pattern background */}
      <div className="absolute inset-0 opacity-15">
        <div className="absolute top-20 left-10 w-32 h-32 border-2 border-blue-400 rounded-full animate-spin-slow shadow-lg shadow-blue-400/20"></div>
        <div className="absolute top-40 right-20 w-24 h-24 border-2 border-slate-400 rounded-full animate-pulse shadow-lg shadow-slate-400/20"></div>
        <div className="absolute bottom-32 left-32 w-40 h-40 border-2 border-indigo-400 rounded-full animate-bounce shadow-lg shadow-indigo-400/20"></div>
        <div className="absolute bottom-20 right-10 w-20 h-20 border-2 border-blue-300 rounded-full animate-ping shadow-lg shadow-blue-300/20"></div>
        <div className="absolute top-1/2 left-1/4 w-16 h-16 border-2 border-slate-300 rounded-full animate-pulse shadow-lg shadow-slate-300/20"></div>
        <div className="absolute top-1/3 right-1/3 w-28 h-28 border-2 border-indigo-300 rounded-full animate-spin-reverse shadow-lg shadow-indigo-300/20"></div>
      </div>

      {/* Enhanced DNA helix pattern */}
      <div className="absolute inset-0 opacity-10">
        <svg
          className="w-full h-full animate-pulse"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          <path
            d="M0,20 Q25,10 50,20 T100,20 M0,30 Q25,40 50,30 T100,30 M0,40 Q25,30 50,40 T100,40 M0,50 Q25,60 50,50 T100,50 M0,60 Q25,50 50,60 T100,60 M0,70 Q25,80 50,70 T100,70"
            stroke="url(#enhancedGradient)"
            strokeWidth="0.8"
            fill="none"
          />
          <defs>
            <linearGradient
              id="enhancedGradient"
              x1="0%"
              y1="0%"
              x2="100%"
              y2="0%"
            >
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
              <stop offset="25%" stopColor="#64748b" stopOpacity="0.5" />
              <stop offset="50%" stopColor="#6366f1" stopOpacity="0.4" />
              <stop offset="75%" stopColor="#64748b" stopOpacity="0.5" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.3" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      {/* Content */}
      <div className="relative z-10 min-h-screen flex items-center">
        <div className="container mx-auto px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Side - About Section */}
            <div className="text-slate-800 space-y-6">
              <div className="space-y-4 relative">
                {/* Enhanced glowing accent line */}
                <div className="absolute -left-6 top-0 w-1 h-32 bg-gradient-to-b from-blue-500 via-indigo-500 to-slate-600 shadow-lg shadow-blue-400/30 animate-pulse"></div>

                {/* Glowing background for title */}
                <div className="absolute -inset-4 bg-gradient-to-r from-blue-200/20 to-slate-200/20 rounded-2xl blur-xl"></div>

                <h1 className="text-4xl lg:text-5xl font-bold leading-tight relative z-10 bg-gradient-to-r from-slate-800 via-blue-700 to-indigo-700 bg-clip-text text-transparent">
                  <Typewriter
                    options={{
                      strings: ["Hi, I'm ViroGen AI 👋"],
                      autoStart: true,
                      loop: true,
                      delay: 75,
                      deleteSpeed: 50,
                    }}
                  />
                </h1>

                <p className="text-slate-600 text-lg leading-relaxed max-w-md relative z-10 drop-shadow-sm">
                  ViroGen is an AI-powered platform designed to predict virus
                  behavior, generate antiviral drug candidates, and evaluate
                  their effectiveness.
                </p>
              </div>
            </div>

            {/* Right Side - Features Section */}
            <div className="space-y-8">
              <h2 className="text-blue-600 text-xl font-semibold mb-8 relative">
                <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Key Capabilities
                </span>
                <div className="absolute -bottom-2 left-0 w-20 h-0.5 bg-gradient-to-r from-blue-500 to-transparent"></div>
              </h2>

              <div className="space-y-6">
                {features.map((feature, index) => {
                  const Icon = feature.icon;
                  return (
                    <Card
                      key={index}
                      className="bg-white/70 border border-blue-200/50 backdrop-blur-sm hover:bg-white/80 hover:border-blue-300/60 transition-all duration-500 hover:scale-105 hover:shadow-xl hover:shadow-blue-200/30 group"
                    >
                      <CardContent className="p-6">
                        <div className="flex items-start space-x-4">
                          <div className="flex-shrink-0">
                            <div className="w-12 h-12 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-lg flex items-center justify-center group-hover:from-blue-200 group-hover:to-indigo-200 transition-all duration-300 shadow-md group-hover:shadow-blue-200/40">
                              <Icon className="w-6 h-6 text-slate-700 group-hover:scale-110 transition-transform duration-300" />
                            </div>
                          </div>
                          <div className="flex-1">
                            <h3 className="text-slate-800 font-semibold text-lg mb-2 group-hover:text-blue-800 transition-colors duration-300">
                              {feature.title}
                            </h3>
                            <p className="text-slate-600 text-sm leading-relaxed group-hover:text-slate-700 transition-colors duration-300">
                              {feature.description}
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced floating elements */}
      <div className="absolute top-1/4 left-1/4 w-3 h-3 bg-blue-400 rounded-full animate-ping opacity-50 shadow-lg shadow-blue-400/30"></div>
      <div className="absolute top-3/4 right-1/4 w-2 h-2 bg-slate-400 rounded-full animate-bounce opacity-50 shadow-lg shadow-slate-400/30"></div>
      <div className="absolute bottom-1/4 left-1/3 w-4 h-4 bg-indigo-400 rounded-full animate-pulse opacity-50 shadow-lg shadow-indigo-400/30"></div>

      <style jsx>{`
        @keyframes float {
          0%,
          100% {
            transform: translateY(0px) rotate(0deg);
          }
          50% {
            transform: translateY(-20px) rotate(180deg);
          }
        }
        @keyframes grid-move {
          0% {
            transform: translate(0, 0);
          }
          100% {
            transform: translate(50px, 50px);
          }
        }
        @keyframes spin-slow {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
        @keyframes spin-reverse {
          from {
            transform: rotate(360deg);
          }
          to {
            transform: rotate(0deg);
          }
        }
        .animate-spin-slow {
          animation: spin-slow 8s linear infinite;
        }
        .animate-spin-reverse {
          animation: spin-reverse 6s linear infinite;
        }
      `}</style>
    </div>
  );
}
