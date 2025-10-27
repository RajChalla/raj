import { Router } from 'express';
import { AuthController } from '@/controllers/authController';
import { requireAuth } from '@/middleware/auth';

const router = Router();
const controller = new AuthController();

router.post('/login', (req, res, next) => controller.login(req, res).catch(next));
router.post('/logout', (req, res, next) => controller.logout(req, res).catch(next));
router.get('/me', requireAuth, (req, res, next) => controller.me(req, res).catch(next));

export default router;
