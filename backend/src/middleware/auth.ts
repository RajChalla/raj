import { Request, Response, NextFunction } from 'express';
import createHttpError from 'http-errors';
import { verifyJwt } from '@/utils/auth';
import { Role } from '@prisma/client';
import { env } from '@/config/env';

declare module 'express-serve-static-core' {
  interface Request {
    user?: {
      id: string;
      role: Role;
      displayName: string;
    };
  }
}

export function authenticate(req: Request, _res: Response, next: NextFunction) {
  const token = req.cookies?.[env.cookieName];
  if (!token) {
    return next();
  }
  try {
    const payload = verifyJwt(token);
    req.user = { id: payload.sub, role: payload.role, displayName: payload.displayName };
  } catch (err) {
    console.warn('Invalid JWT', err);
  }
  next();
}

export function requireAuth(req: Request, _res: Response, next: NextFunction) {
  if (!req.user) {
    throw new createHttpError.Unauthorized();
  }
  next();
}

export function requireRole(...roles: Role[]) {
  return (req: Request, _res: Response, next: NextFunction) => {
    if (!req.user) {
      throw new createHttpError.Unauthorized();
    }
    if (!roles.includes(req.user.role)) {
      throw new createHttpError.Forbidden();
    }
    next();
  };
}
