# Frontend Dockerfile
FROM node:18-alpine

# Установка curl для healthcheck
RUN apk add --no-cache curl

WORKDIR /app

# Копирование package.json и package-lock.json
COPY package*.json ./

# Установка всех зависимостей (включая devDependencies для сборки)
RUN npm ci

# Копирование исходного кода
COPY . .

# Сборка приложения
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]