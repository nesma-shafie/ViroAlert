// "use client"

// import { motion } from "framer-motion"
// import Link from "next/link"
// import {
//   Brush,
//   Dna,
//   ShieldCheck,
//   Brain,
//   Newspaper,
// } from "lucide-react"

// const features = [
//   {
//     title: "ViroGen",
//     description: "Generate antiviral candidates using AI",
//     icon: <Brain className="w-8 h-8" />,
//     href: "/virogen",
//   },
//   {
//     title: "DTI",
//     description: "Predict virus-drug interactions",
//     icon: <ShieldCheck className="w-8 h-8" />,
//     href: "/dti",
//   },
//   {
//     title: "Alignment",
//     description: "Compare virus genome sequences",
//     icon: <Dna className="w-8 h-8" />,
//     href: "/alignment",
//   },
//   {
//     title: "Visualization",
//     description: "Explore data visually and interactively",
//     icon: <Brush className="w-8 h-8" />,
//     href: "/visualization",
//   },
//   {
//     title: "Report",
//     description: "Generate summaries and reports",
//     icon: <Newspaper className="w-8 h-8" />,
//     href: "/report",
//   },
// ]

// export default function FeaturesPage() {
//   return (
//     <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-gray-100 to-blue-100 py-16 px-4 text-center">
//       <motion.h1
//         initial={{ opacity: 0, y: -30 }}
//         animate={{ opacity: 1, y: 0 }}
//         transition={{ duration: 0.6 }}
//         className="text-4xl font-bold text-gray-800 mb-8"
//       >
//         Key Features of ViroGen Platform
//       </motion.h1>

//       <div className="relative w-full max-w-5xl">
//         <motion.div
//           initial={{ scale: 0.8, opacity: 0 }}
//           animate={{ scale: 1, opacity: 1 }}
//           transition={{ duration: 0.6, delay: 0.3 }}
//           className="bg-cyan-600 text-white rounded-full w-52 h-52 flex items-center justify-center mx-auto font-semibold text-lg shadow-lg z-10 relative"
//         >
//           ViroGen Core
//         </motion.div>

//         <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8 mt-16">
//           {features.map((feature, index) => (
//             <motion.div
//               key={index}
//               initial={{ opacity: 0, scale: 0.7 }}
//               animate={{ opacity: 1, scale: 1 }}
//               transition={{ duration: 0.5, delay: 0.4 + index * 0.1 }}
//             >
//               <Link href={feature.href} className="group block">
//                 <div className="flex flex-col items-center bg-white shadow-md p-6 rounded-full w-44 h-44 mx-auto transition-all duration-300 group-hover:scale-110 group-hover:ring-2 group-hover:ring-cyan-600">
//                   <div className="bg-cyan-100 p-4 rounded-full text-cyan-600">
//                     {feature.icon}
//                   </div>
//                   <p className="text-md font-bold text-gray-800 mt-3">{feature.title}</p>
//                   <p className="text-xs text-gray-500 opacity-0 group-hover:opacity-100 mt-1 transition-opacity duration-300">
//                     {feature.description}
//                   </p>
//                 </div>
//               </Link>
//             </motion.div>
//           ))}
//         </div>
//       </div>
//     </div>
//   )
// }


// "use client"

// import { motion, AnimatePresence } from "framer-motion"
// import { useState } from "react"
// import Link from "next/link"
// import { Button } from "@/components/ui/button"
// import { Badge } from "@/components/ui/badge"
// import { Card, CardContent } from "@/components/ui/card"
// import { Brush, Dna, ShieldCheck, Brain, Newspaper, ArrowRight, Sparkles } from "lucide-react"

// const features = [
//   {
//     title: "ViroGen",
//     icon: Brain,
//     description: "Generate novel antiviral candidates using advanced deep learning algorithms and molecular modeling.",
//     detailedDescription:
//       "Our AI-powered ViroGen system leverages cutting-edge machine learning to discover and design new antiviral compounds with high efficacy potential.",
//     href: "/virogen",
//     color: "from-purple-500 to-pink-500",
//     bgColor: "bg-purple-100",
//     textColor: "text-purple-700",
//     x:  "-22rem",
//     y: "-12rem",
//   },
//   {
//     title: "DTI",
//     icon: ShieldCheck,
//     description: "Predict drug-target interactions for known antivirals with unprecedented accuracy.",
//     detailedDescription:
//       "Advanced computational models analyze molecular interactions to predict how drugs will bind to viral targets, optimizing therapeutic outcomes.",
//     href: "/dti",
//     color: "from-green-500 to-emerald-500",
//     bgColor: "bg-green-100",
//     textColor: "text-green-700",
//     x:  "22rem",
//     y:  "-12rem",
//   },
//   {
//     title: "Alignment",
//     icon: Dna,
//     description: "Analyze viral protein similarities and evolutionary patterns across sequences.",
//     detailedDescription:
//       "Comprehensive sequence alignment tools reveal evolutionary relationships and identify conserved regions for targeted drug development.",
//     href: "/alignment",
//     color: "from-blue-500 to-cyan-500",
//     bgColor: "bg-blue-100",
//     textColor: "text-blue-700",
//     x:  "-22rem",
//     y: "12rem",
//   },
//   {
//     title: "Visualization",
//     icon: Brush,
//     description: "Visualize complex interaction networks and compound properties interactively.",
//     detailedDescription:
//       "Interactive 3D visualizations and network graphs make complex molecular data accessible and actionable for researchers.",
//     href: "/visualization",
//     color: "from-orange-500 to-red-500",
//     bgColor: "bg-orange-100",
//     textColor: "text-orange-700",
//     x:  "22rem",
//     y:  "12rem",
//   },
//   {
//     title: "Report",
//     icon: Newspaper,
//     description: "Generate comprehensive reports on antiviral candidate evaluation and analysis.",
//     detailedDescription:
//       "Automated reporting system compiles research findings into professional, publication-ready documents with detailed analytics.",
//     href: "/report",
//     color: "from-indigo-500 to-purple-500",
//     bgColor: "bg-indigo-100",
//     textColor: "text-indigo-700",
//     x: "0rem",
//     y:  "24rem",
//   },
// ]

// export default function FeaturesPage() {
//   const [hovered, setHovered] = useState<number | null>(null)
//   const [isMobile, setIsMobile] = useState(false)

//   // Check if mobile on component mount
//   useState(() => {
//     const checkMobile = () => setIsMobile(window.innerWidth < 768)
//     checkMobile()
//     window.addEventListener("resize", checkMobile)
//     return () => window.removeEventListener("resize", checkMobile)
//   })

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-gray-100 via-blue-100 relative overflow-hidden">
//       {/* Background Effects */}
//       <div className="absolute inset-0">
//         <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
//         <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
//       </div>

//       {/* Header */}
//       <div className="relative z-10 pt-24 pb-12 text-center px-4">
//         <motion.div initial={{ opacity: 0, y: -30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
//           <Badge variant="outline" className="mb-4 border-black text-black-400 bg-black-400">
//             <Sparkles className="w-4 h-4 mr-2" />
//             AI-Powered Platform
//           </Badge>
//           <h1 className="text-4xl md:text-6xl font-bold text-back mb-6">
//             ViroGen
//             <span className="bg-gradient-to-r from-cyan-500 to-blue-500 bg-clip-text text-transparent"> Platform</span>
//           </h1>
//           <p className="text-xl text-black-300 max-w-3xl mx-auto">
//             Revolutionary AI platform for virus analysis, drug discovery, and rapid antiviral development.
//           </p>
//         </motion.div>
//       </div>

//       {/* Interactive Feature Map */}
//       <div className="relative flex items-center justify-center min-h-[600px] md:min-h-[800px] px-4">
//         {/* Core Center */}
//         <motion.div
//           layout
//           initial={{ scale: 0, opacity: 0 }}
//           animate={{ scale: 1, opacity: 1 }}
//           transition={{ duration: 0.8, delay: 0.3 }}
//           className="relative w-44 h-44 md:w-56 md:h-56 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold text-lg md:text-xl shadow-2xl z-20 border-4 border-white/20"
//         >
//           <div className="text-center">
//             <Brain className="w-8 h-8 md:w-10 md:h-10 mx-auto mb-2 " />
//             <div>ViroGen</div>
//             <div className="text-sm font-normal opacity-80">Core</div>
//           </div>

//           {/* Pulsing rings */}
//           <div className="absolute inset-0 rounded-full border-2 border-cyan-400/30 animate-ping"></div>
//           <div className="absolute inset-0 rounded-full border border-cyan-400/20 animate-pulse"></div>
//         </motion.div>

//         {/* Feature Bubbles */}
//         {features.map((feature, index) => {
//           const isActive = hovered === index
//           const Icon = feature.icon
//           const xPos = feature.x
//           const yPos = feature.y

//           return (
//             <motion.div
//               key={index}
//               initial={{ x: xPos, y: yPos, opacity: 0, scale: 0.5 }}
//               animate={{
//                 x: isActive ? "0rem" : xPos,
//                 y: isActive ? "0rem" : yPos,
//                 scale: isActive ? 1.1 : 1,
//                 opacity: 1,
//                 zIndex: isActive ? 30 : 10,
//               }}
//               transition={{
//                 type: "spring",
//                 stiffness: 150,
//                 damping: 20,
//                 delay: 0.5 + index * 0.1,
//               }}
//               onMouseEnter={() => setHovered(index)}
//               onMouseLeave={() => setHovered(null)}
//               className="absolute cursor-pointer"
//             >
//               <Card
//                 className={`w-36 h-36 md:w-44 md:h-44 ${feature.bgColor} border-2 border-white/20 shadow-xl hover:shadow-2xl transition-all duration-300`}
//               >
//                 <CardContent className="flex flex-col items-center justify-center h-full text-center p-4">
//                   <div className={`p-3 rounded-full mb-2 bg-gradient-to-br ${feature.color}`}>
//                     <Icon className="w-6 h-6 md:w-8 md:h-8 text-white" />
//                   </div>
//                   <h3 className={`text-sm md:text-base font-bold ${feature.textColor} mb-1`}>{feature.title}</h3>

//                   <AnimatePresence>
//                     {isActive && (
//                       <motion.div
//                         initial={{ opacity: 0, y: 10 }}
//                         animate={{ opacity: 1, y: 0 }}
//                         exit={{ opacity: 0, y: -10 }}
//                         className="absolute inset-0 bg-white/95 backdrop-blur-sm rounded-lg p-4 flex flex-col justify-center"
//                       >
//                         <p className="text-xs text-gray-700 mb-3 leading-relaxed">{feature.description}</p>
//                         <Link href={feature.href}>
//                           <Button size="sm" className={`bg-gradient-to-r ${feature.color} text-white border-0 group`}>
//                             Explore
//                             <ArrowRight className="w-3 h-3 ml-1 group-hover:translate-x-1 transition-transform" />
//                           </Button>
//                         </Link>
//                       </motion.div>
//                     )}
//                   </AnimatePresence>
//                 </CardContent>
//               </Card>

//               {/* Connection Lines */}
//               {!isActive && (
//                 <motion.div
//                   initial={{ opacity: 0 }}
//                   animate={{ opacity: 0.3 }}
//                   transition={{ delay: 1 + index * 0.1 }}
//                   className="absolute top-1/2 left-1/2 w-1 bg-gradient-to-r from-cyan-400/50 to-transparent"
//                   style={{
//                     height:
//                       Math.sqrt(Math.pow(Number.parseFloat(xPos), 2) + Math.pow(Number.parseFloat(yPos), 2)) / 4 +
//                       "rem",
//                     transformOrigin: "top",
//                     transform: `translate(-50%, -50%) rotate(${(Math.atan2(Number.parseFloat(yPos), Number.parseFloat(xPos)) * 180) / Math.PI}deg)`,
//                   }}
//                 />
//               )}
//             </motion.div>
//           )
//         })}
//       </div>

//       {/* Bottom CTA Section */}
//       <div className="relative z-10 py-20 text-center px-4">
//         <motion.div
//           initial={{ opacity: 0, y: 30 }}
//           animate={{ opacity: 1, y: 0 }}
//           transition={{ duration: 0.8, delay: 1.5 }}
//           className="max-w-4xl mx-auto"
//         >
//           <h2 className="text-3xl md:text-4xl font-bold text-black mb-6">Ready to Accelerate Your Research?</h2>
//           <p className="text-xl text-gray-700 mb-8">
//             Join leading researchers using ViroGen to discover more and more
//           </p>
//         </motion.div>
//       </div>
//     </div>
//   )
// }



"use client"

import { motion, AnimatePresence } from "framer-motion"
import { useEffect, useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Brush, Dna, ShieldCheck, Brain, Newspaper, ArrowRight, Sparkles } from "lucide-react"

const features = [
  {
    title: "ViroGen",
    icon: Brain,
    description: "Generate novel antiviral candidates using advanced deep learning algorithms and molecular modeling.",
    detailedDescription:
      "Our AI-powered ViroGen system leverages cutting-edge machine learning to discover and design new antiviral compounds with high efficacy potential.",
    href: "/virogen",
    color: "from-purple-500 to-pink-500",
    bgColor: "bg-purple-100",
    textColor: "text-purple-700",
    x:  "-22rem",
    y:  "-12rem",
  },
  {
    title: "DTI",
    icon: ShieldCheck,
    description: "Predict drug-target interactions for known antivirals with unprecedented accuracy.",
    detailedDescription:
      "Advanced computational models analyze molecular interactions to predict how drugs will bind to viral targets, optimizing therapeutic outcomes.",
    href: "/dti",
    color: "from-green-500 to-emerald-500",
    bgColor: "bg-green-100",
    textColor: "text-green-700",
    x: "22rem",
    y:  "-12rem",
  },
  {
    title: "Alignment",
    icon: Dna,
    description: "Analyze viral protein similarities and evolutionary patterns across sequences.",
    detailedDescription:
      "Comprehensive sequence alignment tools reveal evolutionary relationships and identify conserved regions for targeted drug development.",
    href: "/alignment",
    color: "from-sky-500 via-cyan-500 to-teal-500",
    bgColor: "bg-cyan-100",
    textColor: "text-cyan-800",
    x: "-22rem",
    y: "12rem",
  },
  {
    title: "Visualization",
    icon: Brush,
    description: "Visualize complex interaction networks and compound properties interactively.",
    detailedDescription:
      "Interactive 3D visualizations and network graphs make complex molecular data accessible and actionable for researchers.",
    href: "/visualization",
    color: "from-orange-500 to-red-500",
    bgColor: "bg-orange-100",
    textColor: "text-orange-700",
    x:  "22rem",
    y:  "12rem",
  },
  {
    title: "Report",
    icon: Newspaper,
    description: "Generate comprehensive reports on antiviral candidate evaluation and analysis.",
    detailedDescription:
      "Automated reporting system compiles research findings into professional, publication-ready documents with detailed analytics.",
    href: "/report",
    color: "from-pink-800 to-rose-500",
      bgColor: "bg-rose-100",
    textColor: "text-rose-700",
    x:  "0rem",
    y: "24rem",
  },
]

export default function FeaturesPage() {
  const [hovered, setHovered] = useState<number | null>(null)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768)
    checkMobile()
    window.addEventListener("resize", checkMobile)
    return () => window.removeEventListener("resize", checkMobile)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-100 via-blue-100 to-slate-200 relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      {/* Header */}
      <div className="relative z-10 pt-24 pb-12 text-center px-4">
        <motion.div initial={{ opacity: 0, y: -30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
          <Badge variant="outline" className="mb-4 border-cyan-600 text-cyan-700 bg-cyan-50">
            <Sparkles className="w-4 h-4 mr-2" />
            AI-Powered Platform
          </Badge>
          <h1 className="text-4xl md:text-7xl font-extrabold text-gray-900 mb-8">
            ViroGen{" "}
            <span className="bg-gradient-to-r from-cyan-500 to-blue-600 bg-clip-text text-transparent">Platform</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Revolutionary AI platform for virus analysis, drug discovery, and rapid antiviral development.
          </p>
        </motion.div>
      </div>

      {/* Interactive Feature Map */}
      <div className="relative flex items-center justify-center min-h-[600px] md:min-h-[800px] px-4">
        {/* Core Center */}
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="relative w-40 h-40 md:w-56 md:h-56 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold text-lg md:text-xl shadow-2xl z-20 border-4 border-white/30"
        >
          <div className="text-center">
            <Brain className="w-10 h-10 md:w-16 md:h-16 mx-auto mb-2" />
            <div>ViroGen</div>
            <div className="text-sm font-normal opacity-80">Core</div>
          </div>

          {/* Pulsing rings */}
          <div className="absolute inset-0 rounded-full border-2 border-cyan-400/30 animate-ping"></div>
          <div className="absolute inset-0 rounded-full border border-cyan-400/20 animate-pulse"></div>
        </motion.div>

        {/* Feature Bubbles */}
        {features.map((feature, index) => {
          const isActive = hovered === index
          const Icon = feature.icon
          const xPos = feature.x
          const yPos = feature.y

          return (
            <motion.div
              key={index}
              initial={{ x: xPos, y: yPos, opacity: 0, scale: 0.5 }}
              animate={{
                x: xPos,
                y: yPos,
                scale: isActive ? 1.1 : 1,
                opacity: 1,
                zIndex: isActive ? 30 : 10,
              }}
              transition={{
                type: "spring",
                stiffness: 150,
                damping: 20,
                delay: 0.5 + index * 0.1,
              }}
              onMouseEnter={() => setHovered(index)}
              onMouseLeave={() => setHovered(null)}
              className="absolute cursor-pointer"
            >
              <Card
                className={`w-32 h-32 md:w-44 md:h-44 ${feature.bgColor} border-2 border-white/40 shadow-xl hover:shadow-2xl transition-all duration-300 relative overflow-hidden`}
              >
                <CardContent className="flex flex-col items-center justify-center h-full text-center p-3 md:p-4">
                  <motion.div
                    animate={{ scale: isActive ? 1.1 : 1 }}
                    transition={{ duration: 0.2 }}
                    className={`p-2 md:p-3 rounded-full mb-2 bg-gradient-to-br ${feature.color}`}
                  >
                    <Icon className="w-5 h-5 md:w-8 md:h-8 text-white" />
                  </motion.div>
                  <h3 className={`text-xs md:text-base font-bold ${feature.textColor} mb-1`}>{feature.title}</h3>

                  {!isActive && (
                    <motion.p
                      className="text-xs text-gray-600 opacity-70"
                      initial={{ opacity: 0.7 }}
                      animate={{ opacity: isActive ? 0 : 0.7 }}
                    >
                      Click to explore
                    </motion.p>
                  )}
                </CardContent>

                {/* Hover Overlay */}
                <AnimatePresence>
                  {isActive && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.8 }}
                      transition={{ duration: 0.2 }}
                      className="absolute inset-0 bg-white/95 backdrop-blur-sm rounded-lg p-3 md:p-4 flex flex-col justify-center"
                    >
                      <p className="text-xs md:text-sm text-gray-700 mb-3 leading-relaxed line-clamp-3">
                        {feature.description}
                      </p>
                      <Link href={feature.href}>
                        <Button
                          size="sm"
                          className={`w-full bg-gradient-to-r ${feature.color} text-white border-0 group hover:shadow-lg transition-all`}
                        >
                          Explore
                          <ArrowRight className="w-3 h-3 ml-1 group-hover:translate-x-1 transition-transform" />
                        </Button>
                      </Link>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Connection Lines */}
                {!isMobile && (
                  <motion.div
                    initial={{ opacity: 0, scaleX: 0 }}
                    animate={{ opacity: 0.2, scaleX: 1 }}
                    transition={{ delay: 1 + index * 0.1, duration: 0.5 }}
                    className="absolute top-1/2 left-1/2 origin-left"
                    style={{
                      width:
                        Math.sqrt(
                          Math.pow(Number.parseFloat(xPos.replace("rem", "")) * 16, 2) +
                            Math.pow(Number.parseFloat(yPos.replace("rem", "")) * 16, 2),
                        ) / 2,
                      height: "2px",
                      background: "linear-gradient(90deg, rgba(34, 197, 94, 0.5) 0%, transparent 100%)",
                      transform: `translate(-50%, -50%) rotate(${
                        Math.atan2(
                          Number.parseFloat(yPos.replace("rem", "")),
                          Number.parseFloat(xPos.replace("rem", "")),
                        ) *
                        (180 / Math.PI)
                      }deg)`,
                    }}
                  />
                )}
              </Card>
            </motion.div>
          )
        })}
      </div>

      {/* Bottom CTA */}
      <div className="relative z-10 py-20 text-center px-4">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.5 }}
          className="max-w-4xl mx-auto"
        >
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">Ready to Accelerate Your Research?</h2>
          <p className="text-xl text-gray-700 mb-8">
            Join leading researchers using ViroGen to discover breakthrough treatments
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white px-8 py-3 text-lg hover:shadow-lg transition-all">
              View Paper
            </Button>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
