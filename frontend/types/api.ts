export interface UploadResponse {
  dataset_id: string;
  quality_issues: Array<Record<string, unknown>>;
  curated_path: string | null;
  preview: Array<Record<string, unknown>>;
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
