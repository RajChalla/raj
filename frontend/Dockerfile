FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json* tsconfig*.json tsconfig.base.json tsconfig.node.json vite.config.ts ./
COPY index.html ./
COPY src ./src
RUN npm install
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --omit=dev
COPY --from=builder /app/dist ./dist
EXPOSE 4173
CMD ["npx", "serve", "-s", "dist", "-l", "4173"]
