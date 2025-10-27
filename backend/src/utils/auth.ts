import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { env } from '@/config/env';
import { Role } from '@prisma/client';

export type JwtPayload = {
  sub: string;
  role: Role;
  displayName: string;
};

export async function hashPassword(password: string): Promise<string> {
  const salt = await bcrypt.genSalt(10);
  return bcrypt.hash(password, salt);
}

export function comparePassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

export function signJwt(payload: JwtPayload): string {
  return jwt.sign(payload, env.jwtSecret, { expiresIn: '8h' });
}

export function verifyJwt(token: string): JwtPayload {
  return jwt.verify(token, env.jwtSecret) as JwtPayload;
}
