import 'dotenv/config';

function requireEnv(name: string, defaultValue?: string): string {
  const value = process.env[name] ?? defaultValue;
  if (!value) {
    throw new Error(`Missing required environment variable ${name}`);
  }
  return value;
}

export const env = {
  port: parseInt(process.env.PORT || '4000', 10),
  databaseUrl: requireEnv('DATABASE_URL'),
  jwtSecret: requireEnv('JWT_SECRET', 'super-secret'),
  nodeEnv: process.env.NODE_ENV || 'development',
  cookieName: process.env.AUTH_COOKIE_NAME || 'fc26_token',
  tickIntervalMs: parseInt(process.env.TICK_INTERVAL_MS || '1000', 10)
};
