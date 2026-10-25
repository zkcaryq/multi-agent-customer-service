import { create } from 'zustand';
import type { ChatState, Message, PendingApproval } from '../types';

interface ChatActions {
  addMessage: (msg: Message) => void;
  appendToLastBotMessage: (chunk: string) => void;
  finishStreaming: (messageId: string, trace?: Record<string, unknown>, content?: string) => void;
  setFeedback: (messageId: string, rating: 'helpful' | 'unhelpful') => void;
  setSessionId: (id: string) => void;
  setConversationId: (id: number) => void;
  setConnectionStatus: (status: ChatState['connectionStatus']) => void;
  setIsBotTyping: (typing: boolean) => void;
  setQueuePosition: (pos: number | null) => void;
  setPendingApproval: (approval: PendingApproval | null) => void;
  clearMessages: () => void;
}

export const useChatStore = create<ChatState & ChatActions>((set) => ({
  messages: [],
  sessionId: null,
  conversationId: 0,
  connectionStatus: 'disconnected',
  isBotTyping: false,
  queuePosition: null,
  pendingApproval: null,

  addMessage: (msg) =>
    set((s) => ({ messages: [...s.messages, msg] })),

  appendToLastBotMessage: (chunk) =>
    set((s) => {
      const msgs = [...s.messages];
      const last = msgs[msgs.length - 1];
      if (last && last.role === 'bot' && last.isStreaming) {
        msgs[msgs.length - 1] = { ...last, content: last.content + chunk };
      }
      return { messages: msgs };
    }),

  finishStreaming: (messageId: string, trace?: Record<string, unknown>, content?: string) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === messageId
          ? { ...m, isStreaming: false, trace, ...(content !== undefined ? { content } : {}) }
          : m
      ),
      isBotTyping: false,
    })),

  setFeedback: (messageId, rating) =>
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === messageId ? { ...m, feedback: rating } : m
      ),
    })),

  setSessionId: (id) => set({ sessionId: id }),
  setConversationId: (id) => set({ conversationId: id }),
  setConnectionStatus: (status) => set({ connectionStatus: status }),
  setIsBotTyping: (typing) => set({ isBotTyping: typing }),
  setQueuePosition: (pos) => set({ queuePosition: pos }),
  setPendingApproval: (approval) => set({ pendingApproval: approval }),
  clearMessages: () => set({ messages: [], pendingApproval: null }),
}));
