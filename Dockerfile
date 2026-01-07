# Dockerfile para RAGproyect
# Imagen base oficial de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requerimientos e instalar dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código fuente
COPY . .

# Exponer el puerto (ajustar si usas otro)
EXPOSE 8000

# Comando para ejecutar la aplicación (ajustar si usas otro entrypoint)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
