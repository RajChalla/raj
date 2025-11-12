import { createHash } from 'crypto';

export function hashPassword(password) {
  return createHash('sha256').update(password).digest('hex');
}

export function verifyPassword(password, hash) {
  return hashPassword(password) === hash;
}
