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

/** LLM model metadata returned from the backend. */
export interface LlmModel {
  id: string;
  name: string;
  provider: string;
  context_length?: number;
  is_default?: boolean;
}

/** Response from the LLM models endpoint. */
export interface LlmModelsResponse {
  default_model: string;
  models: LlmModel[];
}
