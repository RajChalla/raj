import React from 'react';
import { Navbar } from './Navbar';

export const PageLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="page">
    <Navbar />
    <main>{children}</main>
  </div>
);
