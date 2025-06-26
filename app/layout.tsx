import type { Metadata } from 'next';
import './globals.css';
import Navbar from '../components/Navbar';

export const metadata: Metadata = {
  title: 'Virogen',
};

import { ReactNode } from 'react';

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <div className='min-h-screen bg-gradient-to-br from-gray-100 via-blue-100 to-slate-200 p-6'>
          {children}
        </div>
        {/* <Footer /> */}
      </body>
    </html>
  );
}