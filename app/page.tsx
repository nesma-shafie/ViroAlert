"use client"

import { Card, CardContent } from "@/components/ui/card"
import { BrainCog, ShieldCheck, FlaskConical } from "lucide-react"

const iconProps = { className: "w-6 h-6 text-white" }

export default function AboutPage() {
  const features = [
    {
      icon: BrainCog ,
      title: "AI-Driven Prediction",
      description: "Utilizing advanced machine learning and deep learning techniques to predict the pandemic potential of emerging viruses with high accuracy.",
    },
    {
      icon: ShieldCheck,
      title: "Drug Generation",
      description: "Employing structural embeddings and domain-specific fine-tuning to generate novel antiviral candidates that effectively target viral proteins.",
    },
    {
    icon: FlaskConical,
    title: "Biological Relevance",
    description: "Utilizes Drug-Target Interaction (DTI) models to rank generated molecules by biological efficacy and identify top-performing commercial antivirals.",
    }
  ]

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background Image with Overlay */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: "url('/viroGen-bg.jpg')",
          backgroundPosition: "center",
          backgroundSize: "cover",
        }}
      >
        <div className="absolute inset-0 bg-black/20"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 min-h-screen flex items-center">
        <div className="container mx-auto px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Side - About Section */}
            <div className="text-white space-y-6">
              <div className="space-y-4 relative">
                {/* Cyan vertical accent line */}
                <div className="absolute -left-6 top-0 w-1 h-32 bg-cyan-400"></div>
                <h1 className="text-4xl lg:text-5xl font-bold leading-tight">
                  About Our
                  <br />
                  ViroGen AI
                  <br />
                  Platform
                </h1>
                <p className="text-gray-300 text-lg leading-relaxed max-w-md">
                  ViroGen is an AI-powered platform designed to predict virus behavior, generate antiviral drug candidates, and evaluate their effectiveness.
                </p>
              </div>
            </div>

            {/* Right Side - Features Section */}
            <div className="space-y-8">
              <h2 className="text-cyan-400 text-xl font-semibold mb-8">Key Capabilities</h2>

              <div className="space-y-6">
                {features.map((feature, index) => {
                  const Icon = feature.icon
                  return (
                    <Card
                      key={index}
                      className="bg-black/20 border-transparent backdrop-blur-sm"
                    >
                      <CardContent className="p-6">
                        <div className="flex items-start space-x-4">
                          <div className="flex-shrink-0">
                            <div className="w-12 h-12 bg-cyan-500/20 rounded-lg flex items-center justify-center">
                              <Icon className="w-6 h-6 text-white" />
                            </div>
                          </div>
                          <div className="flex-1">
                            <h3 className="text-white font-semibold text-lg mb-2">{feature.title}</h3>
                            <p className="text-gray-300 text-sm leading-relaxed">{feature.description}</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}