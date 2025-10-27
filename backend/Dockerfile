FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json* tsconfig*.json tsconfig.base.json tsconfig.build.json jest.config.ts .eslintrc.cjs ./
COPY prisma ./prisma
RUN npm install
RUN npx prisma generate
COPY src ./src
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --omit=dev
COPY --from=builder /app/dist ./dist
COPY prisma ./prisma
ENV NODE_ENV=production
CMD ["node", "dist/index.js"]
