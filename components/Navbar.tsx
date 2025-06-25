"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();

  type NavLink = {
    href: string;
    label: React.ReactNode;
  };

  const [navLinks, setNavLinks] = useState<NavLink[]>([
    { href: "/", label: "About" },
    { href: "/login", label: "Login" },
  ]);

  const [isAuth, setIsAuth] = useState(() => {
    if (typeof window !== "undefined" && window.localStorage) {
      return localStorage.getItem("auth") === "true";
    }
    return false;
  });

  useEffect(() => {
    if (typeof window !== "undefined" && window.localStorage) {
      const username = localStorage.getItem("username");

      if (isAuth) {
        setNavLinks((prev) => {
          const updated = [...prev];
          const loginIndex = updated.findIndex((link) => link.href === "/login");
          if (loginIndex !== -1) {
            updated.splice(loginIndex, 1, { href: "/features", label: "Features" });
            updated.push({ href: "/logout", label: "Logout" })
            updated.push({
              href: "",
              label: (
                <span className="flex items-center space-x-2 group relative cursor-default select-none">
                  <span className="relative w-[30px] h-[30px] flex items-center justify-center rounded-full bg-gray-200 text-gray-700 font-bold text-lg overflow-hidden">
                    <Image
                      src="/user.png"
                      alt="User Avatar"
                      width={30}
                      height={30}
                      className="rounded-full absolute inset-0 object-cover"
                    />
                    <span className="relative z-10 text-white font-bold">
                      {username && username[0]?.toUpperCase()}
                    </span>
                  </span>
                  <span className="absolute left-1/2 -translate-x-1/2 top-full mt-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                    {username}
                  </span>
                </span>
              ),
            });
          }
          return updated;
        });
      } else {
        setNavLinks((prev) => {
          const updated = [...prev];
          const logoutIndex = updated.findIndex((link) => link.href === "/logout");
          const featureIndex = updated.findIndex((link) => link.href === "/features");
          const profileIndex = updated.findIndex((link) => link.href === "");

          if (logoutIndex !== -1) {
            updated.splice(logoutIndex, 1);
          }

          if (featureIndex !== -1) {
            updated.splice(featureIndex, 1);
          }

          if (profileIndex !== -1) {
            updated.splice(profileIndex, 1);
          }
          if (!updated.find((link) => link.href === "/login")) {
            updated.push({ href: "/login", label: "Login" });
          }
          return updated;
        });
      }
    }
  }, [isAuth, pathname]);

  useEffect(() => {
    if (typeof window !== "undefined" && window.localStorage) {
      setIsAuth(localStorage.getItem("auth") === "true");
    }
  }, [pathname]);

  const handleLogout = () => {
    if (typeof window !== "undefined" && window.localStorage) {
      localStorage.removeItem("auth");
      localStorage.removeItem("username");
      setIsAuth(false);
      router.push("/login");
    }
  };

  return (
    <header className="bg-white text-black shadow-md">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-4 py-1">
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
              key={typeof link.label === "string" ? link.label : link.href}
              href={link.href}
              className={`hover:bg-gray-100 px-3 py-2 rounded-md transition-colors ${pathname === link.href ? "bg-cyan-600 text-white font-semibold" : ""
                }`}
            >
              {link.label}
            </Link>
          ))}

        </nav>
      </div>
    </header>
  );
}