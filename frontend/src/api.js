export const API_BASE = "http://127.0.0.1:8000";

export async function login(username, password) {
  const res = await fetch(`${API_BASE}/api/token/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Login failed");
  }
  return res.json(); // { access, refresh }
}

export async function listDocuments(token) {
  const res = await fetch(`${API_BASE}/api/documents/`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to load documents");
  return res.json();
}

export async function uploadDocument(token, file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/api/documents/`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Upload failed");
  }
  return res.json();
}

export async function queryTfidf(token, docId, question, topK = 5) {
  const res = await fetch(`${API_BASE}/api/query_tfidf/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ doc_id: docId, question, top_k: topK }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Query failed");
  }
  return res.json(); // { answer, hits }
}