"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Icons } from "@/components/icons";

interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
}

interface SidebarProps {
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewChat: () => void;
}

export function Sidebar({ activeSessionId, onSelectSession, onNewChat }: SidebarProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);

  useEffect(() => {
    loadSessions();
    // Listen for storage changes
    const handleStorage = () => loadSessions();
    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  const loadSessions = () => {
    const allSessions: ChatSession[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith("chat_history_")) {
        const sessionId = key.replace("chat_history_", "");
        try {
          const data = JSON.parse(localStorage.getItem(key) || "[]");
          if (data.length > 0) {
            const lastMsg = data[data.length - 1];
            allSessions.push({
              id: sessionId,
              title: sessionId,
              lastMessage: lastMsg.content?.slice(0, 50) || "",
              timestamp: new Date(lastMsg.timestamp),
            });
          }
        } catch (e) {
          console.error("Failed to parse session", key);
        }
      }
    }
    // Sort by most recent
    allSessions.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
    setSessions(allSessions);
  };

  const deleteSession = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    localStorage.removeItem(`chat_history_${sessionId}`);
    loadSessions();
    if (activeSessionId === sessionId) {
      onNewChat();
    }
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Button variant="default" className="w-full gap-2" onClick={onNewChat}>
          {Icons.plus} New Chat
        </Button>
      </div>

      <div className="sidebar-section-label">Recent Chats</div>

      <ScrollArea className="sidebar-list">
        {sessions.length === 0 ? (
          <div className="sidebar-empty">
            No previous chats yet. Upload a dataset to get started.
          </div>
        ) : (
          sessions.map((session) => (
            <button
              key={session.id}
              className={`sidebar-item ${activeSessionId === session.id ? "active" : ""}`}
              onClick={() => onSelectSession(session.id)}
            >
              <div className="sidebar-icon">{Icons.chat}</div>
              <div className="flex flex-col gap-1" style={{ flex: 1, overflow: "hidden" }}>
                <span className="font-medium truncate">{session.title}</span>
                <span className="text-sm truncate" style={{ color: "var(--text-muted)" }}>
                  {session.lastMessage}...
                </span>
              </div>
              <button
                className="sidebar-delete"
                onClick={(e) => deleteSession(e, session.id)}
                title="Delete chat"
              >
                {Icons.close}
              </button>
            </button>
          ))
        )}
      </ScrollArea>

      <div className="sidebar-footer">
        <div className="sidebar-user">
          <div className="sidebar-avatar">{Icons.database}</div>
          <span className="text-sm font-medium">Dataset Curator</span>
        </div>
      </div>
    </aside>
  );
}
