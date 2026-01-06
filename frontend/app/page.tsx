"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { downloadCuratedFile, uploadDataset, sendChatMessage, getPreview } from "@/lib/api";
import type { UploadResponse } from "@/types/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, THead, TBody, TR, TH, TD } from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Sidebar } from "@/components/sidebar";
import { Icons } from "@/components/icons";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

const WELCOME_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content: "Welcome to the **Dataset Curator**!\n\nYou can:\n- **Upload a CSV** using the attachment button\n- **Transform data** - *\"remove column X\"*, *\"filter where Y > 10\"*\n- **Ask questions** about your data\n\nUpload a dataset to get started!",
  timestamp: new Date(),
};

export default function Home() {
  const [datasetId, setDatasetId] = useState("");
  const [preview, setPreview] = useState<UploadResponse["preview"]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalRows, setTotalRows] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE]);
  const [inputMessage, setInputMessage] = useState("");
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [chatWidth, setChatWidth] = useState(50);
  const [isResizing, setIsResizing] = useState(false);
  const [pendingQueue, setPendingQueue] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Handle resize
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !containerRef.current) return;
      
      const container = containerRef.current;
      const rect = container.getBoundingClientRect();
      const newWidth = ((e.clientX - rect.left) / rect.width) * 100;
      
      // Clamp between 30% and 70%
      if (newWidth >= 30 && newWidth <= 70) {
        setChatWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [isResizing]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load chat history and data preview when session changes
  const loadSession = useCallback(async (sessionId: string) => {
    // Load chat history from localStorage
    const savedChat = localStorage.getItem(`chat_history_${sessionId}`);
    if (savedChat) {
      try {
        const parsed = JSON.parse(savedChat) as Array<Omit<ChatMessage, "timestamp"> & { timestamp: string }>;
        setMessages(parsed.map((m) => ({ ...m, timestamp: new Date(m.timestamp) })));
      } catch (e) {
        console.error("Failed to parse chat history", e);
        setMessages([WELCOME_MESSAGE]);
      }
    } else {
      setMessages([WELCOME_MESSAGE]);
    }
    
    setDatasetId(sessionId);
    setActiveSession(sessionId);
    
    // Fetch preview from backend
    try {
      const data = await getPreview(sessionId, 1, 50);
      if (data) {
        setPreview(data.preview || []);
        setCurrentPage(data.page || 1);
        setTotalPages(data.total_pages || 1);
        setTotalRows(data.total_rows || 0);
      } else {
        setPreview([]);
        setCurrentPage(1);
        setTotalPages(1);
        setTotalRows(0);
      }
    } catch (e) {
      console.error("Failed to fetch preview", e);
      setPreview([]);
      setCurrentPage(1);
      setTotalPages(1);
      setTotalRows(0);
    }
  }, []);

  const handleSelectSession = (sessionId: string) => {
    loadSession(sessionId);
  };

  const handleNewChat = () => {
    setDatasetId("");
    setActiveSession(null);
    setPreview([]);
    setCurrentPage(1);
    setTotalPages(1);
    setTotalRows(0);
    setMessages([WELCOME_MESSAGE]);
    setInputMessage("");
  };

  const handlePageChange = async (newPage: number) => {
    if (!datasetId || newPage < 1 || newPage > totalPages) return;
    
    try {
      const data = await getPreview(datasetId, newPage, 50);
      if (data) {
        setPreview(data.preview || []);
        setCurrentPage(data.page || 1);
        setTotalPages(data.total_pages || 1);
        setTotalRows(data.total_rows || 0);
      }
    } catch (e) {
      console.error("Failed to fetch page", e);
    }
  };

  // Update active session when datasetId changes
  useEffect(() => {
    if (datasetId && datasetId !== activeSession) {
      setActiveSession(datasetId);
    }
  }, [datasetId, activeSession]);

  const quickAsk = async (prompt: string) => {
    if (!prompt.trim()) return;
    const sessionId = datasetId || `session_${Date.now()}`;
    if (!datasetId) {
      setDatasetId(sessionId);
      setActiveSession(sessionId);
    }
    addMessage("user", prompt);
    setIsProcessing(true);
    try {
      const response = await sendChatMessage(sessionId, prompt);
      addMessage("assistant", response.assistant_message);
      // Refresh preview
      try {
        const data = await getPreview(sessionId, currentPage, 50);
        if (data) {
          setPreview(data.preview || []);
          setCurrentPage(data.page || 1);
          setTotalPages(data.total_pages || 1);
          setTotalRows(data.total_rows || 0);
        }
      } catch {
        /* ignore */
      }
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      addMessage("assistant", `I encountered an error: ${errorMessage}. Try uploading a dataset first!`);
    } finally {
      setIsProcessing(false);
    }
  };

  const addMessage = (role: "user" | "assistant", content: string, id?: string) => {
    const newMessage: ChatMessage = { 
      id: id || Date.now().toString(), 
      role, 
      content, 
      timestamp: new Date() 
    };
    
    setMessages((prev: ChatMessage[]) => {
      const updated = [...prev, newMessage];
      if (datasetId) {
        localStorage.setItem(`chat_history_${datasetId}`, JSON.stringify(updated));
        // Trigger storage event for sidebar update
        window.dispatchEvent(new Event("storage"));
      }
      return updated;
    });
    
    return newMessage.id;
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    
    // Auto-generate dataset ID from filename
    const autoId = selectedFile.name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9]/g, "_").toLowerCase();
    setDatasetId(autoId);
    setActiveSession(autoId);
    
    // Show upload message
    addMessage("user", `Uploaded: ${selectedFile.name}`);
    
    setIsProcessing(true);
    addMessage("assistant", "Processing your dataset...");

    try {
      const res = (await uploadDataset(autoId, selectedFile)) as UploadResponse;
      const newPreview = res.preview || [];
      
      setPreview(newPreview);
      setCurrentPage(res.page || 1);
      setTotalPages(res.total_pages || 1);
      setTotalRows(res.total_rows || res.row_count || 0);

      // Use actual counts from backend, fallback to preview dimensions only if not available
      const cols = res.column_count !== undefined ? res.column_count : (newPreview[0] ? Object.keys(newPreview[0]).length : 0);
      const rows = res.row_count !== undefined && res.row_count > 0 ? res.row_count : (newPreview.length > 0 ? newPreview.length : 0);

      let summary = `**Dataset loaded!**\n\n`;
      summary += `**${selectedFile.name}**\n`;
      summary += `- ${cols} columns, ${rows} rows\n\n`;
      summary += `You can:\n`;
      summary += `- Say **'show data'** to see the preview\n`;
      summary += `- Say **'remove column X'** to clean it\n`;
      summary += `- Say **'download'** to get the file`;

      addMessage("assistant", summary);
      
      // Save to localStorage for sidebar
      localStorage.setItem(`chat_history_${autoId}`, JSON.stringify([
        WELCOME_MESSAGE,
        { id: Date.now().toString(), role: "user", content: `Uploaded: ${selectedFile.name}`, timestamp: new Date() },
        { id: (Date.now() + 1).toString(), role: "assistant", content: summary, timestamp: new Date() },
      ]));
      window.dispatchEvent(new Event("storage"));
      
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      addMessage("assistant", `Error processing dataset: ${errorMessage}`);
    } finally {
      setIsProcessing(false);
      // Reset file input
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDownload = () => {
    if (!datasetId) {
      addMessage("assistant", "Please process a dataset first before downloading.");
      return;
    }
    addMessage("user", "Download processed dataset");
    addMessage("assistant", "Starting download of your processed dataset...");
    downloadCuratedFile(datasetId);
  };

  const ensureSession = useCallback(() => {
    let sessionId = datasetId;
    if (!sessionId) {
      sessionId = `session_${Date.now()}`;
      setDatasetId(sessionId);
      setActiveSession(sessionId);
    }
    return sessionId;
  }, [datasetId]);

  const runMessage = useCallback(
    async (msg: string) => {
      const sessionId = ensureSession();
      setIsProcessing(true);
      try {
        const response = await sendChatMessage(sessionId, msg);
        addMessage("assistant", response.assistant_message);
        try {
          const data = await getPreview(sessionId, currentPage, 50);
          if (data) {
            setPreview(data.preview || []);
            setCurrentPage(data.page || 1);
            setTotalPages(data.total_pages || 1);
            setTotalRows(data.total_rows || 0);
          }
        } catch {
          /* ignore preview fetch errors */
        }
      } catch (e) {
        const errorMessage = e instanceof Error ? e.message : String(e);
        addMessage("assistant", `I encountered an error: ${errorMessage}. Try uploading a dataset first!`);
      } finally {
        setIsProcessing(false);
      }
    },
    [addMessage, currentPage, ensureSession, getPreview]
  );

  // Process queued messages when idle
  useEffect(() => {
    if (!isProcessing && pendingQueue.length > 0) {
      const next = pendingQueue[0];
      setPendingQueue((prev) => prev.slice(1));
      runMessage(next);
    }
  }, [isProcessing, pendingQueue, runMessage]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const msg = inputMessage.trim();
    const lowerMsg = msg.toLowerCase();
    
    addMessage("user", msg);
    setInputMessage("");

    // Handle special commands locally
    if (lowerMsg.includes("upload") || lowerMsg.includes("attach")) {
      fileInputRef.current?.click();
      return;
    }
    
    if (lowerMsg.includes("download") && preview.length > 0) {
      handleDownload();
      return;
    }

    // Queue if something is already processing or queued
    if (isProcessing || pendingQueue.length > 0) {
      setPendingQueue((prev) => [...prev, msg]);
      return;
    }

    await runMessage(msg);
  };

  return (
    <div className={"app-layout" + (sidebarCollapsed ? " sidebar-collapsed" : "")}>
      <Sidebar
        activeSessionId={activeSession}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
      />
      <button 
        className="toggle-sidebar" 
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        title={sidebarCollapsed ? "Show sidebar" : "Hide sidebar"}
      >
        {sidebarCollapsed ? Icons.chevronRight : Icons.chevronLeft}
      </button>
      <main className="app-main" ref={containerRef}>
        {/* Left: Chat Interface */}
        <div className="chat-panel" style={{ flex: `0 0 ${chatWidth}%` }}>
        <div className="chat-header">
          <h1>Dataset Curator</h1>
          <Badge variant={isProcessing ? "warning" : preview.length ? "success" : "default"}>
            {isProcessing ? "Processing..." : preview.length ? "Ready" : "Idle"}
          </Badge>
        </div>

        <ScrollArea className="chat-messages">
          {messages.map((msg: ChatMessage) => (
            <div key={msg.id} className={`chat-message ${msg.role}`}>
              <Avatar>
                <AvatarFallback>{msg.role === "user" ? Icons.user : Icons.bot}</AvatarFallback>
              </Avatar>
              <div className="message-content">
                <div className="message-header">
                  <span className="message-role">{msg.role === "user" ? "You" : "Curator Agent"}</span>
                  <span className="message-time">
                    {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
                <div className="message-text">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </ScrollArea>

        <div className="chat-input-area">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.json,.parquet"
            onChange={handleFileSelect}
            style={{ display: "none" }}
          />
          <form onSubmit={handleSendMessage} className="message-form">
            <button
              type="button"
              className="attach-button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isProcessing}
              title="Upload CSV file"
            >
              {Icons.paperclip}
            </button>
            <Input
              type="text"
              placeholder={isProcessing ? "Processing..." : "Ask anything or upload a CSV..."}
              value={inputMessage}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setInputMessage(e.target.value)}
            />
            {preview.length > 0 && (
              <button
                type="button"
                className="download-button"
                onClick={handleDownload}
                disabled={isProcessing}
                title="Download processed data"
              >
                {Icons.download}
              </button>
            )}
            <Button type="submit" disabled={!inputMessage.trim()}>
              {Icons.send}
            </Button>
          </form>
        </div>
      </div>

      {/* Resize Handle */}
      <div 
        className={`resize-handle ${isResizing ? "dragging" : ""}`}
        onMouseDown={handleMouseDown}
      />

      {/* Right: Data Preview */}
      <div className="data-panel" style={{ flex: `1 1 ${100 - chatWidth}%` }}>
        <Card>
          <div className="data-header">
            <h2>Data Preview</h2>
            {preview.length > 0 && (
              <div className="data-stats">
                <Badge variant="secondary">{preview[0] ? Object.keys(preview[0]).length : 0} columns</Badge>
                <Badge variant="secondary">{totalRows || preview.length} rows</Badge>
                {totalPages > 1 && (
                  <Badge variant="secondary">Page {currentPage} of {totalPages}</Badge>
                )}
              </div>
            )}
          </div>
          {/* Quick chips */}
          {preview.length > 0 && (
            <div className="quick-actions">
              <span className="quick-label">Try:</span>
              <div className="chips">
                <Button size="sm" variant="outline" onClick={() => quickAsk("Count rows")}>Row count</Button>
                <Button size="sm" variant="outline" onClick={() => quickAsk("List columns")}>List columns</Button>
                <Button size="sm" variant="outline" onClick={() => quickAsk("Group by course and count")}>Count by course</Button>
                <Button size="sm" variant="outline" onClick={() => quickAsk("Average study_hours by course")}>Avg study_hours by course</Button>
                <Button size="sm" variant="outline" onClick={() => quickAsk("Show missing values summary")}>Missing values</Button>
              </div>
            </div>
          )}
          {/* Column chips */}
          {preview.length > 0 && (
            <div className="columns-bar">
              {Object.keys(preview[0]).map((col) => (
                <Badge key={col} variant="secondary" className="column-chip">{col}</Badge>
              ))}
            </div>
          )}
          {preview.length > 0 ? (
            <>
              <ScrollArea className="table-scroll">
                <Table>
                  <THead>
                    <TR>
                      {Object.keys(preview[0]).map((col: string) => (
                        <TH key={col} className="sticky-th">{col}</TH>
                      ))}
                    </TR>
                  </THead>
                  <TBody>
                    {preview.map((row: Record<string, unknown>, idx: number) => (
                      <TR key={idx}>
                        {Object.entries(row).map(([col, val]: [string, unknown]) => (
                          <TD
                            key={col}
                            className={typeof val === "number" || (!isNaN(Number(val)) && val !== null && val !== "") ? "num-cell" : ""}
                          >
                            {val === null || val === "" ? <span className="null-value">null</span> : String(val)}
                          </TD>
                        ))}
                      </TR>
                    ))}
                  </TBody>
                </Table>
              </ScrollArea>
              
              {totalPages > 1 && (
                <div className="pagination-controls" style={{ padding: "1rem", display: "flex", justifyContent: "space-between", alignItems: "center", borderTop: "1px solid #e5e7eb" }}>
                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(currentPage - 1)}
                      disabled={currentPage === 1}
                    >
                      Previous
                    </Button>
                    <span style={{ fontSize: "0.875rem", color: "#6b7280" }}>
                      Page {currentPage} of {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePageChange(currentPage + 1)}
                      disabled={currentPage === totalPages}
                    >
                      Next
                    </Button>
                  </div>
                  <div style={{ fontSize: "0.875rem", color: "#6b7280" }}>
                    Showing {((currentPage - 1) * 50) + 1} - {Math.min(currentPage * 50, totalRows)} of {totalRows} rows
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">{Icons.barChart}</div>
              <h3>No data yet</h3>
              <p>Upload and process a CSV file to see a preview here.</p>
            </div>
          )}
        </Card>
      </div>
      </main>
    </div>
  );
}
