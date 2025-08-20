import type { Metadata } from 'next';
import './globals.css';
import Navbar from '../components/Navbar';
import { FormProvider } from '@/context/FormContext';

export const metadata: Metadata = {
  title: 'Virogen',
  description: 'Virogen - Your app description here',
  icons: {
    icon: '/v.png', // This sets the favicon in the browser tab
  },
 
};

import { ReactNode } from 'react';

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Navbar />
        <div className='min-h-screen bg-gradient-to-br from-gray-100 via-blue-100 to-slate-200 p-6'>
          <FormProvider>
            {children}
          </FormProvider>
        </div>
        {/* <Footer /> */}
      </body>
    </html>
  );
}