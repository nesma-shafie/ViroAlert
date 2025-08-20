import { Facebook, Twitter, Youtube, Github, Linkedin } from 'lucide-react'
import Link from 'next/link'

export default function Footer() {
  return (
    <footer className="bg-gray-100 text-gray-700 text-sm pt-4">
      {/* Follow Section */}
    <div className=" flex justify-center space-x-8">
      <span className="font-bold">FOLLOW VIROGEN</span>
    </div>

      {/* Social Icons Row */}
    <div className="bg-gray-100 text-gray-700 py-4 flex justify-center space-x-20">
        <a href="https://x.com" target="_blank" rel="noopener noreferrer"><Twitter size={32} /></a>
        <a href="https://facebook.com" target="_blank" rel="noopener noreferrer"><Facebook size={32} /></a>
        <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer"><Linkedin size={32} /></a>
        <a href="https://github.com" target="_blank" rel="noopener noreferrer"><Github size={32} /></a>
    </div>

      {/* Footer bar */}
    <div className="bg-virogen-blue py-2 text-white text-center text-xs">
    ViroGen © 2025 | All Rights Reserved
    </div>
    </footer>
  )
}
