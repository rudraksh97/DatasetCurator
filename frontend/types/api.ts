export interface UploadResponse {
  dataset_id: string;
  preview: Array<Record<string, unknown>>;
  row_count?: number;
  column_count?: number;
}

export interface HealthResponse {
  dataset_id: string;
  issues: Array<Record<string, unknown>>;
}

export interface ApproveResponse {
  dataset_id: string;
  curated_path: string | null;
  version: number;
}

export interface DownloadResponse {
  dataset_id: string;
  curated_path: string | null;
}

export interface DatasetCardResponse {
  dataset_id: string;
  dataset_card: Record<string, unknown>;
}
