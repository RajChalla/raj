import { createId } from '../utils/id.js';
import { hashPassword, verifyPassword } from '../utils/password.js';
import { createToken, verifyToken } from '../utils/token.js';
import { findUserByUsername, findUserById, saveUser } from './storeService.js';

export function ensureSeedUsers() {
  if (findUserByUsername('admin')) return;
  const admin = { id: createId(), username: 'admin', passwordHash: hashPassword('admin123'), role: 'ADMIN' };
  const owner1 = { id: createId(), username: 'owner1', passwordHash: hashPassword('owner123'), role: 'OWNER' };
  const owner2 = { id: createId(), username: 'owner2', passwordHash: hashPassword('owner123'), role: 'OWNER' };
  const viewer = { id: createId(), username: 'viewer', passwordHash: hashPassword('viewer123'), role: 'VIEWER' };
  for (const user of [admin, owner1, owner2, viewer]) {
    saveUser(user);
  }
}

export function login({ username, password }) {
  const user = findUserByUsername(username);
  if (!user) {
    throw new Error('Invalid credentials');
  }
  if (!verifyPassword(password, user.passwordHash)) {
    throw new Error('Invalid credentials');
  }
  const token = createToken({ sub: user.id, role: user.role }, 60 * 60 * 8);
  return { token, user: { id: user.id, username: user.username, role: user.role } };
}

export function authenticate(token) {
  const payload = verifyToken(token);
  if (!payload) return null;
  const user = findUserById(payload.sub);
  if (!user) return null;
  return { id: user.id, username: user.username, role: user.role };
}
