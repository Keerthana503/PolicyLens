import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listDocuments, uploadDocument, queryTfidf } from "../api";

function DashboardPage() {
  const [docs, setDocs] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [hits, setHits] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [error, setError] = useState("");

  const navigate = useNavigate();
  const token = localStorage.getItem("accessToken");

  useEffect(() => {
    if (!token) {
      navigate("/login");
      return;
    }

    async function loadDocs() {
      try {
        const data = await listDocuments(token);
        setDocs(data);
        if (data.length > 0) {
          setSelectedDocId(data[0].id);
        }
      } catch (err) {
        console.error(err);
        setError("Failed to load documents");
      }
    }

    loadDocs();
  }, [token, navigate]);

  const handleLogout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("refreshToken");
    navigate("/login");
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file || !token) return;

    setUploading(true);
    setError("");
    try {
      await uploadDocument(token, file);
      const data = await listDocuments(token);
      setDocs(data);
      if (data.length > 0) {
        setSelectedDocId(data[0].id);
      }
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setUploading(false);
      e.target.value = ""; // reset file input
    }
  };

  const handleAsk = async () => {
    if (!token) {
      setError("Not logged in");
      return;
    }
    if (!selectedDocId) {
      setError("Select a document first");
      return;
    }
    if (!question.trim()) {
      setError("Type a question");
      return;
    }

    setQuerying(true);
    setError("");
    setAnswer("");
    setHits([]);
    try {
      const res = await queryTfidf(token, selectedDocId, question.trim(), 5);
      setAnswer(res.answer || "");
      setHits(res.hits || []);
    } catch (err) {
      setError(err.message || "Query failed");
    } finally {
      setQuerying(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="logo">PolicyLens</div>
        <button onClick={handleLogout} className="outline">
          Logout
        </button>
      </header>

      <main className="layout">
        {/* Left: Document & upload */}
        <section className="panel">
          <h2>Documents</h2>
          <div className="field">
            <label className="label">Upload PDF</label>
            <input type="file" accept="application/pdf" onChange={handleUpload} />
            {uploading && <div className="small-text">Uploading...</div>}
          </div>

          <div className="field">
            <label className="label">Select document</label>
            <select
              value={selectedDocId}
              onChange={(e) => setSelectedDocId(e.target.value)}
            >
              {docs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.file?.split("/").slice(-1)[0] || `Document ${d.id}`}
                </option>
              ))}
            </select>
            {docs.length === 0 && (
              <div className="small-text">No documents yet. Upload a PDF.</div>
            )}
          </div>
        </section>

        {/* Right: Question & answer */}
        <section className="panel">
          <h2>Ask your document</h2>
          <div className="field">
            <textarea
              rows={3}
              placeholder="Ask a question about the selected PDF..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>
          <button onClick={handleAsk} disabled={querying}>
            {querying ? "Searching..." : "Ask"}
          </button>

          {error && <div className="error" style={{ marginTop: "1rem" }}>{error}</div>}

          <div className="answer-block">
            <h3>Answer</h3>
            {answer ? <p>{answer}</p> : <p className="muted">No answer yet.</p>}
          </div>

          {hits && hits.length > 0 && (
            <div className="hits-block">
              <h3>Matched snippets</h3>
              <ul>
                {hits.slice(0, 5).map((h, i) => (
                  <li key={i}>
                    <div className="small-text">Page {h.page} — score {h.score.toFixed(3)}</div>
                    <div>{h.text}</div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default DashboardPage;