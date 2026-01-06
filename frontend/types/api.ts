/**
 * API type definitions for the Dataset Curator frontend.
 */

/** Response from the upload endpoint. */
export interface UploadResponse {
  dataset_id: string;
  preview: Array<Record<string, unknown>>;
  row_count?: number;
  column_count?: number;
  page?: number;
  page_size?: number;
  total_rows?: number;
  total_pages?: number;
}

/** Response from the preview endpoint. */
export interface PreviewResponse {
  dataset_id: string;
  preview: Array<Record<string, unknown>>;
  row_count: number;
  column_count: number;
  page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
}

/** Response from the chat endpoint. */
export interface ChatResponse {
  user_message: string;
  assistant_message: string;
}
