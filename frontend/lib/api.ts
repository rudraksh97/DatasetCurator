const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

export async function uploadDataset(datasetId: string, file: File) {
  const form = new FormData();
  form.append("dataset_id", datasetId);
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    body: form,
  });
  return handle(res);
}

export function downloadCuratedFile(datasetId: string) {
  window.location.href = `${BASE_URL}/download/${datasetId}/file`;
}

export async function sendChatMessage(datasetId: string, content: string) {
  const res = await fetch(`${BASE_URL}/chat/${datasetId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  return handle<{ user_message: string; assistant_message: string }>(res);
}

export async function getPreview(datasetId: string, page: number = 1, pageSize: number = 50) {
  const res = await fetch(`${BASE_URL}/preview/${datasetId}?page=${page}&page_size=${pageSize}`);
  if (res.status === 404) return null;
  return handle<{ dataset_id: string; preview: Array<Record<string, any>>; row_count: number; column_count: number; page: number; page_size: number; total_rows: number; total_pages: number }>(res);
}
