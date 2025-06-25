"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Check } from "lucide-react";
import { motion } from "framer-motion";
import Image from "next/image";

const Logout = () => {
  const router = useRouter();

  useEffect(() => {
    // Perform logout
    
    // Redirect to home page after 3 seconds
    const timer = setTimeout(() => {
      router.push("/");
    }, 3000);

    return () => clearTimeout(timer);
  }, [router]);

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="bg-white rounded-lg shadow-lg p-12 text-center max-w-md w-full mx-4 flex flex-col items-center">
        {/* Logo */}
        <div className="mb-8 flex flex-col items-center">
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
        </div>
        
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
            <Check className="w-8 h-8 text-green-600" />
          </div>
        </div>

        {/* Message */}
        <div className="mb-6">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">
            You have been logged out
          </h1>
          <p className="text-gray-600">
            Thank you
          </p>
        </div>

        {/* Footer */}
        <div className="text-xs text-gray-400 uppercase tracking-wide">
          POWERED BY
        </div>
        <div className="text-sm text-gray-500 font-medium mt-1">
          VIROGEN
        </div>
      </div>
  );
};

export default Logout;
