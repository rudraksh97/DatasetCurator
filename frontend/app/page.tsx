"use client";

import { useState, useRef, useEffect, useCallback } from "react";
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
  content: "Welcome to the **Dataset Curator**!\n\nYou can:\n- 🔍 **Search for datasets** - *\"find climate data\"*, *\"search for stock prices\"*\n- 📎 **Upload a CSV** using the attachment button\n- 🔧 **Transform data** - *\"remove column X\"*, *\"filter where Y > 10\"*\n\nWhat data are you looking for?",
  timestamp: new Date(),
};

export default function Home() {
  const [datasetId, setDatasetId] = useState("");
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

  // Load chat history and data preview when session changes
  const loadSession = useCallback(async (sessionId: string) => {
    // Load chat history from localStorage
    const savedChat = localStorage.getItem(`chat_history_${sessionId}`);
    if (savedChat) {
      try {
        const parsed = JSON.parse(savedChat);
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
    
    // Fetch preview from backend
    try {
      const data = await getPreview(sessionId);
      if (data) {
        setPreview(data.preview || []);
        setIssues(data.quality_issues || []);
      } else {
        setPreview([]);
        setIssues([]);
      }
    } catch (e) {
      console.error("Failed to fetch preview", e);
      setPreview([]);
      setIssues([]);
    }
  }, []);

  const handleSelectSession = (sessionId: string) => {
    loadSession(sessionId);
  };

  const handleNewChat = () => {
    setDatasetId("");
    setActiveSession(null);
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

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    
    // Auto-generate dataset ID from filename
    const autoId = selectedFile.name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9]/g, "_").toLowerCase();
    setDatasetId(autoId);
    setActiveSession(autoId);
    
    // Show upload message
    addMessage("user", `📎 Uploaded: ${selectedFile.name}`);
    
    setIsProcessing(true);
    addMessage("assistant", "⏳ Processing your dataset... Analyzing schema and checking data quality.");

    try {
      const res = (await uploadDataset(autoId, selectedFile)) as UploadResponse;
      const newPreview = res.preview || [];
      const newIssues = res.quality_issues || [];
      
      setPreview(newPreview);
      setIssues(newIssues);

      const cols = newPreview[0] ? Object.keys(newPreview[0]).length : 0;
      const rows = newPreview.length;
      const issueCount = newIssues.length;

      let summary = `✅ **Dataset loaded!**\n\n`;
      summary += `📊 **${selectedFile.name}**\n`;
      summary += `- ${cols} columns, ${rows} rows\n`;
      
      if (issueCount > 0) {
        summary += `- ⚠️ ${issueCount} quality issues found\n\n`;
        summary += `You can:\n`;
        summary += `- Say **'show data'** to see the preview\n`;
        summary += `- Say **'remove column X'** to clean it\n`;
        summary += `- Say **'download'** to get the file`;
      } else {
        summary += `- ✨ No quality issues!\n\n`;
        summary += `Your data looks clean! You can ask me to transform it or download it.`;
      }

      addMessage("assistant", summary);
      
      // Save to localStorage for sidebar
      localStorage.setItem(`chat_history_${autoId}`, JSON.stringify([
        WELCOME_MESSAGE,
        { id: Date.now().toString(), role: "user", content: `📎 Uploaded: ${selectedFile.name}`, timestamp: new Date() },
        { id: (Date.now() + 1).toString(), role: "assistant", content: summary, timestamp: new Date() },
      ]));
      window.dispatchEvent(new Event("storage"));
      
    } catch (e: any) {
      addMessage("assistant", `❌ Error processing dataset: ${e.message}`);
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

    // Auto-generate session ID if none exists
    let sessionId = datasetId;
    if (!sessionId) {
      sessionId = `session_${Date.now()}`;
      setDatasetId(sessionId);
      setActiveSession(sessionId);
    }

    setIsProcessing(true);
    try {
      const response = await sendChatMessage(sessionId, msg);
      addMessage("assistant", response.assistant_message);
      
      // Refresh data preview after chat (transformations may have occurred)
      try {
        const data = await getPreview(datasetId);
        if (data) {
          setPreview(data.preview || []);
          setIssues(data.quality_issues || []);
        }
      } catch {
        // Ignore preview fetch errors
      }
    } catch (e: any) {
      addMessage("assistant", `I encountered an error: ${e.message}. Try uploading a dataset first!`);
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
              onChange={(e) => setInputMessage(e.target.value)}
              disabled={isProcessing}
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
