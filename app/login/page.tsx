"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import axios from "axios";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Image from "next/image";
import { motion } from "framer-motion";


import { toast } from "sonner";

export default function AuthPage() {
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();
  const baseURL = process.env.NEXT_PUBLIC_API_BASE_URL;

  const handleSignUp = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    const formData = new FormData(e.currentTarget);
    const email = formData.get("email") as string;
    const password = formData.get("password") as string;
    const confirmPassword = formData.get("confirmPassword") as string;
    const username = formData.get("username") as string;
    if (password !== confirmPassword) {
      toast.error("Signup failed. Please try again.", {
        style: { background: "#fee2e2", color: "#dc2626", borderLeft: "4px solid #dc2626" },
      });
      // alert("Passwords do not match!");
      setIsLoading(false);
      return;
    }

    const body = {
      email,
      password,
      username,
      // Add other signup data as needed (e.g., confirmPassword)
    };
    try {
      const response = await axios.post(
        `${baseURL}/auth/signup`,
        body
        // Add other signup data as needed (e.g., password)
      );
      setIsLoading(false);
      if (response.status === 200) {
        localStorage.setItem("token", response.data.data.token);
        localStorage.setItem("auth", "true");
        localStorage.setItem("username", username);
        router.push("/");
      } else {
        alert("Signup failed. Please try again.");
      }
    } catch (error) {
      setIsLoading(false);
      alert("Signup failed. Please try again.");
    }
    // Optionally handle response, e.g., redirect or show message
  };

  const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    const formData = new FormData(e.currentTarget);
    const username = formData.get("username") as string;
    const password = formData.get("password") as string;

    const body = {
      username,
      password,
    };
    try {
      const response = await axios.post(
        `${baseURL}/auth/login`,
        body
        // Add other signup data as needed (e.g., password)
      );
      setIsLoading(false);

      if (response.status === 200) {
        localStorage.setItem("token", response.data.data.token);
        localStorage.setItem("auth", "true");
        localStorage.setItem("username", username);
        router.push("/");
      } else {
        alert("Login failed. Please check your credentials.");
      }
    } catch {
      setIsLoading(false);

      alert("Login failed. Please check your credentials.");
    }
  };

  return (
    <div className="min-h-[100vh] overflow-hidden flex items-center justify-center bg-gradient-to-br from-gray-100 to-blue-100">
      <div className="max-w-md w-full space-y-6">
        <div className="flex flex-col items-center">
          <motion.div
            animate={{ y: [0, -20, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          >
            <Image
              src="/ViroGen.png"
              alt="ViroGen Logo"
              width={100}
              height={100}
              className="rounded-full border border-gray-300 shadow-md"
            />
          </motion.div>
          <h2 className="mt-4 text-3xl font-extrabold text-gray-900">
            Welcome to ViroGen
          </h2>
          <p className="mt-1 text-sm text-gray-600">
            Sign in or create a new account
          </p>
        </div>

        <Card className="rounded-2xl shadow-xl border border-gray-200">
          <CardContent className="p-6">
            <Tabs defaultValue="login" className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-6">
                <TabsTrigger value="login">Login</TabsTrigger>
                <TabsTrigger value="signup">Sign Up</TabsTrigger>
              </TabsList>

              <TabsContent value="login">
                <form onSubmit={handleLogin} className="space-y-4">
                  <div>
                    <Label htmlFor="login-username">UserName</Label>
                    <Input
                      id="login-username"
                      name="username"
                      type="text"
                      placeholder="Enter your  username"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="login-password">Password</Label>
                    <Input
                      id="login-password"
                      name="password"
                      type="password"
                      placeholder="Enter your password"
                      required
                    />
                  </div>
                  <Button
                    type="submit"
                    className="w-full mt-2 virogen-blue hover:virogen-light-blue"
                    disabled={isLoading}
                  >
                    {isLoading ? "Logging in..." : "Log In"}
                  </Button>
                </form>
              </TabsContent>

              <TabsContent value="signup">
                <form onSubmit={handleSignUp} className="space-y-4">
                  <div>
                    <Label htmlFor="signup-username">UserName</Label>
                    <Input
                      id="signup-username"
                      name="username"
                      type="text"
                      placeholder="Enter your  username"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="signup-email">Email</Label>
                    <Input
                      id="signup-email"
                      name="email"
                      type="email"
                      placeholder="Enter your email"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="signup-password">Password</Label>
                    <Input
                      id="signup-password"
                      name="password"
                      type="password"
                      placeholder="Create a password"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="confirm-password">Confirm Password</Label>
                    <Input
                      id="confirm-password"
                      name="confirmPassword"
                      type="password"
                      placeholder="Confirm your password"
                      required
                    />
                  </div>
                  <Button
                    type="submit"
                    className="w-full mt-2 virogen-blue hover:virogen-light-blue"
                    disabled={isLoading}
                  >
                    {isLoading ? "Creating account..." : "Sign Up"}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
