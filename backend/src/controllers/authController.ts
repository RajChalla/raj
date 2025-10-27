import { Request, Response } from 'express';
import createHttpError from 'http-errors';
import { prisma } from '@/config/prisma';
import { comparePassword, signJwt } from '@/utils/auth';
import { env } from '@/config/env';

export class AuthController {
  async login(req: Request, res: Response) {
    const { email, password } = req.body;
    if (!email || !password) {
      throw new createHttpError.BadRequest('Missing credentials');
    }
    const user = await prisma.user.findUnique({ where: { email } });
    if (!user) {
      throw new createHttpError.Unauthorized('Invalid credentials');
    }
    const valid = await comparePassword(password, user.passwordHash);
    if (!valid) {
      throw new createHttpError.Unauthorized('Invalid credentials');
    }
    const token = signJwt({ sub: user.id, role: user.role, displayName: user.displayName });
    res.cookie(env.cookieName, token, { httpOnly: true, sameSite: 'lax' });
    res.json({ id: user.id, email: user.email, role: user.role, displayName: user.displayName });
  }

  async logout(_req: Request, res: Response) {
    res.clearCookie(env.cookieName);
    res.status(204).send();
  }

  async me(req: Request, res: Response) {
    if (!req.user) {
      return res.status(204).send();
    }
    const user = await prisma.user.findUnique({ where: { id: req.user.id } });
    if (!user) {
      return res.status(204).send();
    }
    res.json({ id: user.id, email: user.email, role: user.role, displayName: user.displayName });
  }
}
