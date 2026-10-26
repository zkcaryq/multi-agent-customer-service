import { create } from 'zustand';
import type { AdminUser } from '../types';

const TOKEN_KEY = 'ma-cs-admin-token';
const API_BASE = 'http://localhost:8000/api/v1';

interface AuthState {
  user: AdminUser | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
}

// 刷新页面后 localStorage 里的令牌还在，直接恢复登录态；令牌是否仍然有效交给
// 接口裁决——admin.ts 收到 401 会清令牌并跳回登录页。
const storedToken = localStorage.getItem(TOKEN_KEY);

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: storedToken,
  isAuthenticated: !!storedToken,

  login: async (username, password) => {
    if (!username) return false;
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      // 后端对"用户不存在"和"密码错误"给同一个 401，这里不再细分
      if (!res.ok) return false;
      const data = await res.json();
      localStorage.setItem(TOKEN_KEY, data.access_token);
      set({ isAuthenticated: true, token: data.access_token, user: data.user });
      return true;
    } catch {
      return false;
    }
  },

  logout: () => {
    localStorage.removeItem(TOKEN_KEY);
    set({ user: null, token: null, isAuthenticated: false });
  },
}));
