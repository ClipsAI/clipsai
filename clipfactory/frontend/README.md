# Clip Factory Frontend

React/Next.js frontend for the Clip Factory viral clip generation system.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Components](#components)
- [Hooks](#hooks)
- [API Integration](#api-integration)
- [State Management](#state-management)
- [Styling](#styling)
- [TypeScript](#typescript)
- [Development](#development)
- [Build & Deployment](#build--deployment)

## Overview

The Clip Factory frontend is a modern web application built with Next.js 14, TypeScript, and Tailwind CSS. It provides an intuitive interface for uploading videos, managing clips, generating variations, and scheduling posts.

**Key Features**:
- Drag-and-drop video upload
- Real-time processing status
- Video preview with timeline scrubbing
- Variation generation interface
- Title A/B testing
- Posting calendar
- Voice-based title input

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Frontend Architecture                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐    ┌──────────────┐             │
│  │  Next.js App │───▶│  API Routes  │             │
│  │  Directory   │    │  (Optional)  │             │
│  └──────┬───────┘    └──────────────┘             │
│         │                                           │
│  ┌──────▼────────────────────────────────────┐    │
│  │          Components Layer                 │    │
│  │  ┌────────────┐  ┌────────────────────┐  │    │
│  │  │   Pages    │  │    UI Components   │  │    │
│  │  │            │  │   (shadcn/Radix)   │  │    │
│  │  └────────────┘  └────────────────────┘  │    │
│  └───────────────────────────────────────────┘    │
│         │                      │                   │
│  ┌──────▼──────────────────────▼──────────┐       │
│  │         Custom Hooks Layer            │       │
│  │  - useVideoUpload                     │       │
│  │  - useProcessingStatus                │       │
│  │  - useVariations                      │       │
│  └───────────────────────────────────────┘       │
│         │                                          │
│  ┌──────▼──────────────────────────────────┐     │
│  │      API Client (Axios + SWR)          │     │
│  └───────────────┬────────────────────────┘     │
│                  │                                │
└──────────────────┼────────────────────────────────┘
                   │
                   ▼
           ┌───────────────┐
           │ Backend API   │
           │  (FastAPI)    │
           └───────────────┘
```

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| Next.js | 14.1.0 | React framework with App Directory |
| React | 18.2.0 | UI library |
| TypeScript | 5.x | Type safety |
| Tailwind CSS | 3.3.0 | Utility-first styling |
| shadcn/ui | Latest | Pre-built components |
| Radix UI | Latest | Headless UI primitives |
| Axios | 1.6.5 | HTTP client |
| SWR | 2.2.4 | Data fetching & caching |
| React Dropzone | 14.2.3 | File upload |
| React Speech Recognition | 3.10.0 | Voice input |
| Framer Motion | 11.0.3 | Animations |
| React Icons | 5.0.1 | Icon library |

## Getting Started

### Prerequisites

- Node.js 18.x or higher
- npm or yarn
- Backend API running on `http://localhost:8000`

### Installation

```bash
cd clipfactory/frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Environment Variables

Create `.env.local`:

```env
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000/api

# Optional: Analytics
NEXT_PUBLIC_GA_ID=G-XXXXXXXXXX
```

## Project Structure

```
frontend/
├── app/                      # Next.js 14 App Directory
│   ├── page.tsx              # Home page
│   ├── layout.tsx            # Root layout
│   ├── globals.css           # Global styles
│   ├── upload/               # Upload interface
│   │   └── page.tsx
│   ├── dashboard/            # Clip management
│   │   └── page.tsx
│   ├── variations/           # Variation generator
│   │   └── page.tsx
│   └── calendar/             # Posting calendar
│       └── page.tsx
├── components/               # React components
│   ├── UploadInterface.tsx   # Main upload UI
│   ├── VariationGenerator.tsx # Variation generation UI
│   ├── PostingHelper.tsx     # Posting management
│   ├── ui/                   # shadcn/ui components
│   │   ├── button.tsx
│   │   ├── dialog.tsx
│   │   ├── slider.tsx
│   │   └── tabs.tsx
│   └── ...
├── hooks/                    # Custom React hooks
│   ├── useVideoUpload.ts
│   ├── useProcessingStatus.ts
│   └── useVariations.ts
├── lib/                      # Utilities
│   ├── api.ts                # API client
│   ├── utils.ts              # Helper functions
│   └── types.ts              # TypeScript types
├── public/                   # Static assets
│   ├── images/
│   └── icons/
├── styles/                   # Additional styles
├── package.json              # Dependencies
├── tsconfig.json             # TypeScript config
├── tailwind.config.js        # Tailwind config
└── next.config.js            # Next.js config
```

## Components

### UploadInterface.tsx

Main video upload component with drag-and-drop support.

**Props**: None (standalone component)

**State**:
```typescript
interface UploadState {
  file: File | null;
  uploading: boolean;
  progress: number;
  videoId: string | null;
  error: string | null;
}
```

**Features**:
- Drag-and-drop file upload
- File validation (format, size)
- Upload progress bar
- Error handling
- Automatic redirect after upload

**Usage**:
```tsx
import UploadInterface from '@/components/UploadInterface';

export default function UploadPage() {
  return <UploadInterface />;
}
```

---

### VariationGenerator.tsx

Interface for generating and previewing clip variations.

**Props**:
```typescript
interface VariationGeneratorProps {
  clipId: string;
  initialClipData?: ClipMetadata;
}
```

**Features**:
- Temporal variation selection (base, +4s, +35s)
- Reframe style selection (original, flipped, blurry_bg)
- Music track selection
- Title style selection
- Real-time preview (future)
- Batch generation

**Usage**:
```tsx
import VariationGenerator from '@/components/VariationGenerator';

export default function VariationsPage({ params }) {
  return <VariationGenerator clipId={params.clipId} />;
}
```

---

### PostingHelper.tsx

Posting calendar and title generation.

**Props**:
```typescript
interface PostingHelperProps {
  variationId: string;
}
```

**Features**:
- Screenshot upload
- Title generation with account type selection
- Calendar integration
- Posting schedule
- Account selection
- Title A/B testing interface

**Usage**:
```tsx
import PostingHelper from '@/components/PostingHelper';

export default function PostingPage({ params }) {
  return <PostingHelper variationId={params.variationId} />;
}
```

---

### UI Components (shadcn/ui)

Pre-built components from shadcn/ui:

```tsx
import { Button } from '@/components/ui/button';
import { Dialog } from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import { Tabs } from '@/components/ui/tabs';

// Usage
<Button variant="default" size="lg">
  Upload Video
</Button>
```

Available components:
- Button
- Dialog / Modal
- Slider
- Tabs
- Toast / Alert
- Card
- Input
- Select
- Checkbox
- Radio Group

## Hooks

### useVideoUpload

Handle video file uploads.

```typescript
import { useVideoUpload } from '@/hooks/useVideoUpload';

function Component() {
  const { upload, uploading, progress, error } = useVideoUpload();

  const handleUpload = async (file: File) => {
    const result = await upload(file);
    console.log('Video ID:', result.videoId);
  };

  return (
    <div>
      {uploading && <Progress value={progress} />}
      {error && <Alert>{error}</Alert>}
    </div>
  );
}
```

**API**:
```typescript
interface UseVideoUpload {
  upload: (file: File) => Promise<VideoUploadResponse>;
  uploading: boolean;
  progress: number;
  error: string | null;
}
```

---

### useProcessingStatus

Poll for video processing status.

```typescript
import { useProcessingStatus } from '@/hooks/useProcessingStatus';

function Component({ videoId }) {
  const { status, progress, clipsFound, isLoading } = useProcessingStatus(videoId);

  return (
    <div>
      <p>Status: {status}</p>
      <p>Progress: {progress * 100}%</p>
      <p>Clips Found: {clipsFound}</p>
    </div>
  );
}
```

**API**:
```typescript
interface UseProcessingStatus {
  status: 'processing' | 'completed' | 'failed';
  progress: number;
  clipsFound: number;
  isLoading: boolean;
  error: Error | null;
}
```

---

### useVariations

Fetch and manage variations.

```typescript
import { useVariations } from '@/hooks/useVariations';

function Component({ clipId }) {
  const { variations, generate, isLoading } = useVariations(clipId);

  const handleGenerate = async () => {
    await generate({
      temporal: ['base', '+4s', '+35s'],
      reframe: ['original', 'flipped']
    });
  };

  return (
    <div>
      {variations.map(v => (
        <VariationCard key={v.id} variation={v} />
      ))}
      <Button onClick={handleGenerate}>Generate</Button>
    </div>
  );
}
```

**API**:
```typescript
interface UseVariations {
  variations: Variation[];
  generate: (config: VariationConfig) => Promise<void>;
  isLoading: boolean;
  error: Error | null;
}
```

## API Integration

### API Client (`lib/api.ts`)

Axios-based API client with interceptors.

```typescript
import axios from 'axios';

const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor (add auth token, etc.)
apiClient.interceptors.request.use((config) => {
  // Add API key if available
  const apiKey = localStorage.getItem('api_key');
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey;
  }
  return config;
});

// Response interceptor (error handling)
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle errors globally
    if (error.response?.status === 401) {
      // Redirect to login
    }
    return Promise.reject(error);
  }
);

export default apiClient;
```

### API Methods

```typescript
// Upload video
export async function uploadVideo(file: File): Promise<VideoUploadResponse> {
  const formData = new FormData();
  formData.append('video', file);

  const response = await apiClient.post('/phase1/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return response.data;
}

// Get processing status
export async function getProcessingStatus(videoId: string) {
  const response = await apiClient.get(`/phase1/status/${videoId}`);
  return response.data;
}

// Generate variations
export async function generateVariations(config: VariationRequest) {
  const response = await apiClient.post('/phase4/generate-variations', config);
  return response.data;
}
```

### SWR Integration

Use SWR for data fetching with caching:

```typescript
import useSWR from 'swr';
import { getProcessingStatus } from '@/lib/api';

function Component({ videoId }) {
  const { data, error, isLoading } = useSWR(
    `/phase1/status/${videoId}`,
    () => getProcessingStatus(videoId),
    {
      refreshInterval: 5000, // Poll every 5 seconds
      revalidateOnFocus: false,
    }
  );

  if (isLoading) return <Spinner />;
  if (error) return <Error message={error.message} />;

  return <StatusDisplay status={data.status} />;
}
```

## State Management

### Local Component State

Use React hooks for component-local state:

```typescript
import { useState } from 'react';

function Component() {
  const [selected, setSelected] = useState<string[]>([]);

  return (
    <Checkbox
      checked={selected.includes('item')}
      onCheckedChange={() => {
        setSelected(prev =>
          prev.includes('item')
            ? prev.filter(i => i !== 'item')
            : [...prev, 'item']
        );
      }}
    />
  );
}
```

### Global State (Future)

For complex global state, consider:
- **Zustand** (lightweight)
- **React Context** (built-in)
- **Redux Toolkit** (if needed)

Example with Zustand:

```typescript
// store/useStore.ts
import create from 'zustand';

interface AppState {
  videoId: string | null;
  setVideoId: (id: string) => void;
}

export const useStore = create<AppState>((set) => ({
  videoId: null,
  setVideoId: (id) => set({ videoId: id }),
}));

// Usage in component
import { useStore } from '@/store/useStore';

function Component() {
  const { videoId, setVideoId } = useStore();
  // ...
}
```

## Styling

### Tailwind CSS

Utility-first CSS framework:

```tsx
<div className="flex items-center justify-between p-4 bg-gray-100 rounded-lg">
  <h2 className="text-2xl font-bold text-gray-900">Title</h2>
  <Button className="bg-blue-500 hover:bg-blue-600">Action</Button>
</div>
```

### Custom Styles

For complex styling, use CSS modules or styled-components:

```tsx
// styles/component.module.css
.container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
}

// Component
import styles from '@/styles/component.module.css';

function Component() {
  return <div className={styles.container}>...</div>;
}
```

### Theme Configuration

Customize Tailwind in `tailwind.config.js`:

```javascript
module.exports = {
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f0f9ff',
          500: '#0ea5e9',
          900: '#0c4a6e',
        },
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
};
```

## TypeScript

### Type Definitions (`lib/types.ts`)

```typescript
export interface VideoUploadResponse {
  video_id: string;
  filename: string;
  size: number;
  duration?: number;
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
}

export interface ClipMetadata {
  clip_id: string;
  start_time: number;
  end_time: number;
  duration: number;
  hook_score?: number;
  transcript?: string;
}

export interface Variation {
  variation_id: string;
  temporal: 'base' | '+4s' | '+35s';
  reframe: 'original' | 'flipped' | 'blurry_bg';
  status: 'pending' | 'processing' | 'completed';
}

export interface TitleVariant {
  variant_id: string;
  text: string;
  hook_style: string;
  predicted_ctr: number;
}
```

### Component Props Types

```typescript
interface ComponentProps {
  videoId: string;
  onComplete?: (result: Result) => void;
  className?: string;
  children?: React.ReactNode;
}

const Component: React.FC<ComponentProps> = ({ videoId, onComplete, className }) => {
  // ...
};
```

## Development

### Running Development Server

```bash
npm run dev
```

Access at `http://localhost:3000`

### Linting

```bash
npm run lint
```

### Type Checking

```bash
npx tsc --noEmit
```

### Formatting

```bash
npm run format
```

### Testing (Future)

```bash
npm test
```

## Build & Deployment

### Production Build

```bash
npm run build
npm start
```

### Environment-Specific Builds

```bash
# Production
NEXT_PUBLIC_API_URL=https://api.clipfactory.io npm run build

# Staging
NEXT_PUBLIC_API_URL=https://staging-api.clipfactory.io npm run build
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --production

COPY . .
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
```

Build and run:

```bash
docker build -t clipfactory-frontend .
docker run -p 3000:3000 clipfactory-frontend
```

### Vercel Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Production deployment
vercel --prod
```

---

## Troubleshooting

### Common Issues

**Issue**: API requests fail with CORS error

**Solution**: Ensure backend has CORS configured:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Issue**: Environment variables not loading

**Solution**: Restart dev server after changing `.env.local`

**Issue**: TypeScript errors in components

**Solution**: Run `npm run lint` and fix type errors

---

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

## Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [shadcn/ui Documentation](https://ui.shadcn.com)
- [TypeScript Documentation](https://www.typescriptlang.org/docs)

---

**Last Updated**: 2025-11-10

For frontend questions, open a GitHub issue with the `frontend` label.
