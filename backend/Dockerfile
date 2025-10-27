FROM node:18-alpine
WORKDIR /app
COPY backend/package.json ./package.json
RUN npm set fund false && npm set audit false
# no dependencies required
COPY backend/src ./src
COPY backend/data ./data
COPY web ./web
ENV NODE_ENV=production
EXPOSE 4000
CMD ["node", "src/server.js"]
