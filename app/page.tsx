// "use client"

// import { useState } from "react"
// import { useRouter } from "next/navigation"
// import { Button } from "@/components/ui/button"
// import { Input } from "@/components/ui/input"
// import { Label } from "@/components/ui/label"
// import {
//   Card,
//   CardContent,
//   CardDescription,
//   CardHeader,
//   CardTitle,
// } from "@/components/ui/card"
// import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
// import Image from "next/image"

// export default function AuthPage() {
//   const [isLoading, setIsLoading] = useState(false)
//   const router = useRouter()

//   const handleSignUp = async (e: React.FormEvent<HTMLFormElement>) => {
//     e.preventDefault()
//     setIsLoading(true)
//     const formData = new FormData(e.currentTarget)
//     const email = formData.get("email") as string
//     const password = formData.get("password") as string
//     const confirmPassword = formData.get("confirmPassword") as string
//     if (password !== confirmPassword) {
//       alert("Passwords do not match!")
//       setIsLoading(false)
//       return
//     }
//     setTimeout(() => {
//       localStorage.setItem("user", JSON.stringify({ email, authenticated: true }))
//       router.push("/ViroGen/app/auth/login")
//       setIsLoading(false)
//     }, 1000)
//   }

//   const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
//     e.preventDefault()
//     setIsLoading(true)
//     const formData = new FormData(e.currentTarget)
//     const email = formData.get("email") as string
//     const password = formData.get("password") as string
//     setTimeout(() => {
//       localStorage.setItem("user", JSON.stringify({ email, authenticated: true }))
//       router.push("/hello")
//       setIsLoading(false)
//     }, 1000)
//   }

//   return (
//     <div className="h-screen overflow-hidden flex items-center justify-center bg-gradient-to-br from-gray-50 to-blue-100">
//       <div className="max-w-md w-full space-y-6">
//         <div className="flex flex-col items-center">
//           <Image
//             src="/ViroGen.png"
//             alt="ViroGen Logo"
//             width={100}
//             height={100}
//             className="rounded-full border border-gray-300 shadow-md"
//           />
//           <h2 className="mt-4 text-3xl font-extrabold text-gray-900">Welcome to ViroGen</h2>
//           <p className="mt-1 text-sm text-gray-600">Sign in or create a new account</p>
//         </div>

//         <Card className="rounded-2xl shadow-xl border border-gray-200">
//           <CardContent className="p-6">
//             <Tabs defaultValue="login" className="w-full">
//               <TabsList className="grid w-full grid-cols-2 mb-6">
//                 <TabsTrigger value="login">Login</TabsTrigger>
//                 <TabsTrigger value="signup">Sign Up</TabsTrigger>
//               </TabsList>

//               <TabsContent value="login">
//                 <form onSubmit={handleLogin} className="space-y-4">
//                   <div>
//                     <Label htmlFor="login-email">Email</Label>
//                     <Input
//                       id="login-email"
//                       name="email"
//                       type="email"
//                       placeholder="Enter your email"
//                       required
//                     />
//                   </div>
//                   <div>
//                     <Label htmlFor="login-password">Password</Label>
//                     <Input
//                       id="login-password"
//                       name="password"
//                       type="password"
//                       placeholder="Enter your password"
//                       required
//                     />
//                   </div>
//                   <Button type="submit" className="w-full mt-2" disabled={isLoading}>
//                     {isLoading ? "Logging in..." : "Log In"}
//                   </Button>
//                 </form>
//               </TabsContent>

//               <TabsContent value="signup">
//                 <form onSubmit={handleSignUp} className="space-y-4">
//                   <div>
//                     <Label htmlFor="signup-email">Email</Label>
//                     <Input
//                       id="signup-email"
//                       name="email"
//                       type="email"
//                       placeholder="Enter your email"
//                       required
//                     />
//                   </div>
//                   <div>
//                     <Label htmlFor="signup-password">Password</Label>
//                     <Input
//                       id="signup-password"
//                       name="password"
//                       type="password"
//                       placeholder="Create a password"
//                       required
//                     />
//                   </div>
//                   <div>
//                     <Label htmlFor="confirm-password">Confirm Password</Label>
//                     <Input
//                       id="confirm-password"
//                       name="confirmPassword"
//                       type="password"
//                       placeholder="Confirm your password"
//                       required
//                     />
//                   </div>
//                   <Button type="submit" className="w-full mt-2" disabled={isLoading}>
//                     {isLoading ? "Creating account..." : "Sign Up"}
//                   </Button>
//                 </form>
//               </TabsContent>
//             </Tabs>
//           </CardContent>
//         </Card>
//       </div>
//     </div>
//   )
// }




"use client"

import { Card, CardContent } from "@/components/ui/card"
import { BrainCog, ShieldCheck, FlaskConical } from "lucide-react"

export default function AboutPage() {
  const features = [
    {
      icon: BrainCog,
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
                  ViroGen is a cutting-edge AI platform developed to generate virus-targeted antiviral candidates using
                  deep learning, structural embeddings, and domain-specific fine-tuning.
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
                              <Icon className="w-6 h-6 text-cyan-500" />
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