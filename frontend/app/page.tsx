"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { downloadCuratedFile, uploadDataset, getChatHistory, sendChatMessage } from "@/lib/api";
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
  content: "Welcome to the Agentic Dataset Curator! Upload a CSV file to get started. I'll analyze it and help you clean your data.",
  timestamp: new Date(),
};

export default function Home() {
  const [datasetId, setDatasetId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<UploadResponse["preview"]>([]);
  const [issues, setIssues] = useState<UploadResponse["quality_issues"]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE]);
  const [inputMessage, setInputMessage] = useState("");
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [chatWidth, setChatWidth] = useState(50);
  const [isResizing, setIsResizing] = useState(false);
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

  // Load chat history when session changes
  const loadSession = useCallback((sessionId: string) => {
    const saved = localStorage.getItem(`chat_history_${sessionId}`);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setMessages(parsed.map((m: any) => ({ ...m, timestamp: new Date(m.timestamp) })));
      } catch (e) {
        console.error("Failed to parse chat history", e);
        setMessages([WELCOME_MESSAGE]);
      }
    } else {
      setMessages([WELCOME_MESSAGE]);
    }
    setDatasetId(sessionId);
    setActiveSession(sessionId);
    // Clear file and preview when switching sessions
    setFile(null);
    setPreview([]);
    setIssues([]);
  }, []);

  const handleSelectSession = (sessionId: string) => {
    loadSession(sessionId);
  };

  const handleNewChat = () => {
    setDatasetId("");
    setActiveSession(null);
    setFile(null);
    setPreview([]);
    setIssues([]);
    setMessages([WELCOME_MESSAGE]);
    setInputMessage("");
  };

  // Update active session when datasetId changes
  useEffect(() => {
    if (datasetId && datasetId !== activeSession) {
      setActiveSession(datasetId);
    }
  }, [datasetId, activeSession]);

  const addMessage = (role: "user" | "assistant", content: string) => {
    const newMessage: ChatMessage = { 
      id: Date.now().toString(), 
      role, 
      content, 
      timestamp: new Date() 
    };
    
    setMessages((prev) => {
      const updated = [...prev, newMessage];
      if (datasetId) {
        localStorage.setItem(`chat_history_${datasetId}`, JSON.stringify(updated));
        // Trigger storage event for sidebar update
        window.dispatchEvent(new Event("storage"));
      }
      return updated;
    });
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      addMessage("user", `Selected file: ${selectedFile.name}`);
      addMessage("assistant", `Great! I see you've selected "${selectedFile.name}". Enter a dataset ID and click "Process Dataset" to start the analysis.`);
    }
  };

  const handleUpload = async () => {
    if (!file || !datasetId) {
      addMessage("assistant", "Please select a file and enter a dataset ID first.");
      return;
    }

    setIsProcessing(true);
    await addMessage("user", `Process dataset "${datasetId}" with file "${file.name}"`);
    await addMessage("assistant", "Starting analysis... I'm ingesting your dataset, inferring the schema, and checking data quality.");

    try {
      const res = (await uploadDataset(datasetId, file)) as UploadResponse;
      setPreview(res.preview || []);
      setIssues(res.quality_issues || []);

      const cols = res.preview?.[0] ? Object.keys(res.preview[0]).length : 0;
      const rows = res.preview?.length || 0;
      const issueCount = res.quality_issues?.length || 0;

      let summary = `✅ Processing complete!\n\n`;
      summary += `📊 **Dataset Summary:**\n`;
      summary += `• Columns: ${cols}\n`;
      summary += `• Preview rows: ${rows}\n`;
      summary += `• Quality issues found: ${issueCount}\n\n`;

      if (issueCount > 0) {
        summary += `⚠️ **Issues detected:**\n`;
        res.quality_issues?.slice(0, 5).forEach((issue: any, i: number) => {
          summary += `${i + 1}. ${issue.column || "General"}: ${issue.issue || issue.description || "Issue detected"} (${issue.severity || "medium"})\n`;
        });
        if (issueCount > 5) {
          summary += `...and ${issueCount - 5} more issues.\n`;
        }
        summary += `\nI've applied automatic fixes where possible. Click "Download Processed" to get the cleaned dataset.`;
      } else {
        summary += `🎉 No quality issues detected! Your data looks clean. Click "Download Processed" to get the dataset.`;
      }

      await addMessage("assistant", summary);
    } catch (e: any) {
      await addMessage("assistant", `❌ Error processing dataset: ${e.message}`);
    } finally {
      setIsProcessing(false);
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

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;

    const msg = inputMessage.trim();
    const lowerMsg = msg.toLowerCase();
    
    addMessage("user", msg);
    setInputMessage("");

    // Handle special commands locally first
    if (lowerMsg.includes("upload") || lowerMsg.includes("file")) {
      fileInputRef.current?.click();
      addMessage("assistant", "Click 'Choose File' to select a CSV file to upload.");
      return;
    }
    
    if (lowerMsg.includes("download") && preview.length > 0) {
      handleDownload();
      return;
    }
    
    if ((lowerMsg.includes("process") || lowerMsg.includes("analyze")) && file && datasetId) {
      handleUpload();
      return;
    }

    // For all other messages, use the LLM-powered backend
    if (!datasetId) {
      addMessage("assistant", "Please enter a dataset ID first so I can provide context-aware assistance. Upload a file to get started!");
      return;
    }

    setIsProcessing(true);
    try {
      const response = await sendChatMessage(datasetId, msg);
      addMessage("assistant", response.assistant_message);
    } catch (e: any) {
      addMessage("assistant", `I encountered an error: ${e.message}. Please make sure a dataset is uploaded first.`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className={`app-layout ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
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
          {messages.map((msg) => (
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
          <div className="upload-controls">
            <Input
              type="text"
              placeholder="Dataset ID (e.g., my-dataset)"
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              disabled={isProcessing}
            />
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.json,.parquet"
              onChange={handleFileSelect}
              style={{ display: "none" }}
            />
            <Button variant="secondary" className="gap-2" onClick={() => fileInputRef.current?.click()} disabled={isProcessing}>
              {Icons.file} {file ? file.name.slice(0, 16) : "Choose File"}
            </Button>
          </div>
          <div className="action-buttons">
            <Button className="gap-2" onClick={handleUpload} disabled={isProcessing || !file || !datasetId}>
              {isProcessing ? Icons.processing : Icons.sparkle} {isProcessing ? "Processing..." : "Process Dataset"}
            </Button>
            <Button variant="secondary" className="gap-2" onClick={handleDownload} disabled={isProcessing || !preview.length}>
              {Icons.download} Download
            </Button>
          </div>
          <form onSubmit={handleSendMessage} className="message-form">
            <Input
              type="text"
              placeholder="Ask me anything about your data..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              disabled={isProcessing}
            />
            <Button type="submit" disabled={isProcessing || !inputMessage.trim()}>
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
                <Badge variant="secondary">{preview.length} rows</Badge>
                <Badge variant={issues.length > 0 ? "warning" : "success"}>
                  {issues.length} issues
                </Badge>
              </div>
            )}
          </div>

          {preview.length > 0 ? (
            <ScrollArea className="table-scroll">
              <Table>
                <THead>
                  <TR>
                    {Object.keys(preview[0]).map((col) => (
                      <TH key={col}>{col}</TH>
                    ))}
                  </TR>
                </THead>
                <TBody>
                  {preview.map((row, idx) => (
                    <TR key={idx}>
                      {Object.entries(row).map(([col, val]) => (
                        <TD key={col}>{val === null || val === "" ? <span className="null-value">null</span> : String(val)}</TD>
                      ))}
                    </TR>
                  ))}
                </TBody>
              </Table>
            </ScrollArea>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">{Icons.barChart}</div>
              <h3>No data yet</h3>
              <p>Upload and process a CSV file to see a preview here.</p>
            </div>
          )}
        </Card>

        {issues.length > 0 && (
          <Card>
            <h2>Quality Issues</h2>
            <ScrollArea className="issues-scroll">
              {issues.map((issue: any, idx: number) => (
                <div key={idx} className="issue-item">
                  <Badge variant={issue.severity === "high" ? "destructive" : issue.severity === "medium" ? "warning" : "secondary"}>
                    {issue.severity || "medium"}
                  </Badge>
                  <div className="issue-details">
                    <span className="issue-column">{issue.column || "General"}</span>
                    <span className="issue-desc">{issue.issue || issue.description}</span>
                  </div>
                </div>
              ))}
            </ScrollArea>
          </Card>
        )}
      </div>
      </main>
    </div>
  );
}
