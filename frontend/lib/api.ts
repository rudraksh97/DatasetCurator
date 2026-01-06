/**
 * API client for the Dataset Curator backend.
 * 
 * All functions communicate with the FastAPI backend for dataset
 * operations including upload, preview, download, and chat.
 */

import type { UploadResponse, PreviewResponse, ChatResponse } from "@/types/api";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

/**
 * Handle API response and throw on error.
 * 
 * @param res - Fetch response object.
 * @returns Parsed JSON response.
 * @throws Error if response is not ok.
 */
async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

/**
 * Upload a dataset file to the backend.
 * 
 * @param datasetId - Unique identifier for the dataset.
 * @param file - CSV file to upload.
 * @returns Upload response with preview data.
 */
export async function uploadDataset(datasetId: string, file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("dataset_id", datasetId);
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    body: form,
  });
  return handle<UploadResponse>(res);
}

/**
 * Trigger download of the processed dataset file.
 * 
 * @param datasetId - Unique identifier for the dataset.
 */
export function downloadCuratedFile(datasetId: string): void {
  window.location.href = `${BASE_URL}/download/${datasetId}/file`;
}

/**
 * Send a chat message to the dataset curator agent.
 * 
 * @param datasetId - Unique identifier for the dataset.
 * @param content - Message content.
 * @returns Chat response with user and assistant messages.
 */
export async function sendChatMessage(datasetId: string, content: string): Promise<ChatResponse> {
  const res = await fetch(`${BASE_URL}/chat/${datasetId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  return handle<ChatResponse>(res);
}

/**
 * Get paginated preview of a dataset.
 * 
 * @param datasetId - Unique identifier for the dataset.
 * @param page - Page number (1-indexed).
 * @param pageSize - Number of rows per page.
 * @returns Preview response with data and pagination, or null if not found.
 */
export async function getPreview(
  datasetId: string, 
  page: number = 1, 
  pageSize: number = 50
): Promise<PreviewResponse | null> {
  const res = await fetch(`${BASE_URL}/preview/${datasetId}?page=${page}&page_size=${pageSize}`);
  if (res.status === 404) return null;
  return handle<PreviewResponse>(res);
}
