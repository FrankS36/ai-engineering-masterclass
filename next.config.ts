import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  env: {
    API_KEY: process.env.GEMINI_API_KEY || process.env.API_KEY || '',
    GEMINI_API_KEY: process.env.GEMINI_API_KEY || process.env.API_KEY || '',
  },
};

export default nextConfig;
