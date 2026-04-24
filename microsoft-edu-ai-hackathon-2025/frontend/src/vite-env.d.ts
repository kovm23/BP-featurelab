/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_UPLOAD_ENDPOINT?: string;
  readonly VITE_API_BASE?: string;
  readonly VITE_DEMO_MODE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}