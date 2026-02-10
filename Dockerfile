FROM node:20-alpine AS build
WORKDIR /app
COPY web/package*.json ./
RUN npm ci
COPY web/ .
RUN npm run build

FROM node:20-alpine
RUN npm install -g serve@14
COPY --from=build /app/dist /app
EXPOSE 3000
CMD ["serve", "-s", "/app", "-l", "3000"]
