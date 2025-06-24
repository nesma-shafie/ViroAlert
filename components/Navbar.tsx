"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

  const navLinks = [
    { href: "/", label: "About" },
    { href: "/contact", label: "Contact" },
    { href: "/login", label: "Login" },
  ];

  return (
    <header className="bg-gray-10 text-black shadow-md">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4">
        {/* Logo + Brand */}
        <Link href="/" className="flex items-center space-x-2">
          <Image
            src="/ViroGen.png"
            alt="ViroGen Logo"
            width={50}
            height={50}
            className="rounded-full"
          />
          <span className="text-xl font-bold tracking-tight">ViroGen</span>
        </Link>

        <nav className="flex space-x-6 ml-auto text-lg">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}              className={`hover:bg-gray-200 px-3 py-2 rounded-md transition-colors ${
                pathname === link.href
                  ? "bg-virogen-blue text-white font-semibold"
                  : ""
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>
        <div className="hidden sm:flex space-x-6">
          {/* Add nav links or buttons here if needed */}
        </div>
      </div>
    </header>
  );
}
