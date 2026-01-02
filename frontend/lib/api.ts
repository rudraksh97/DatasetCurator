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

export async function getHealth(datasetId: string) {
  const res = await fetch(`${BASE_URL}/health/${datasetId}`);
  return handle(res);
}

export async function approveFixes(datasetId: string, fixes: Array<Record<string, unknown>>) {
  const res = await fetch(`${BASE_URL}/approve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId, fixes }),
  });
  return handle(res);
}

export async function getCard(datasetId: string) {
  const res = await fetch(`${BASE_URL}/card/${datasetId}`);
  return handle(res);
}

export async function downloadCurated(datasetId: string) {
  const res = await fetch(`${BASE_URL}/download/${datasetId}`);
  return handle(res);
}

export function downloadCuratedFile(datasetId: string) {
  // Trigger browser download
  window.location.href = `${BASE_URL}/download/${datasetId}/file`;
}

export async function getChatHistory(datasetId: string) {
  const res = await fetch(`${BASE_URL}/chat/${datasetId}`);
  if (res.status === 404) return [];
  return handle<Array<{ role: string; content: string; timestamp: string }>>(res);
}

export async function sendChatMessage(datasetId: string, content: string) {
  const res = await fetch(`${BASE_URL}/chat/${datasetId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  return handle<{ user_message: string; assistant_message: string; history: Array<any> }>(res);
}

export async function analyzeDataset(datasetId: string) {
  const res = await fetch(`${BASE_URL}/analyze/${datasetId}`, {
    method: "POST",
  });
  return handle<{ dataset_id: string; analysis: string }>(res);
}
