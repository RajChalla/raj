process.env.DATABASE_URL = process.env.DATABASE_URL || 'postgresql://user:pass@localhost:5432/test';
process.env.JWT_SECRET = 'test-secret';
process.env.AUTH_COOKIE_NAME = 'test_token';

jest.setTimeout(30000);
