// app/components/Navbar.tsx
import Image from 'next/image';
import Link from 'next/link';

export default function Navbar() {
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
            <Link href="/about" className="hover:underline">
                About
            </Link>
            <Link href="/contact" className="hover:underline">
                Contact
            </Link>
            <Link href="/login" className="hover:underline">
                Login
            </Link>
        </nav>
        <div className="hidden sm:flex space-x-6">
          {/* Add nav links or buttons here if needed */}
        </div>
      </div>
    </header>
  );
}
