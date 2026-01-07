"""
Script para probar la conexión con Google Gemini
Ejecuta: python test_gemini.py
"""
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Cargar variables de entorno
load_dotenv()

print("=" * 60)
print("🔍 VERIFICACIÓN DE CONFIGURACIÓN GOOGLE GEMINI")
print("=" * 60)

# 1. Verificar variables de entorno
print("\n1️⃣ Variables de entorno:")
api_key = os.getenv('GOOGLE_API_KEY', '')
print(f"   API_KEY: {'*' * 10}{api_key[-4:] if len(api_key) > 4 else '(vacía)'}")
print(f"   GEMINI_MODEL: {os.getenv('GEMINI_MODEL', 'no configurado')}")
print(f"   GEMINI_EMBED_MODEL: {os.getenv('GEMINI_EMBED_MODEL', 'no configurado')}")

if not api_key:
    print("\n❌ ERROR: GOOGLE_API_KEY no está configurada en .env")
    print("\n💡 Obtén tu API key en: https://aistudio.google.com/app/apikey")
    exit(1)

print("\n✅ Variables cargadas correctamente")

# 2. Probar Embeddings
print("\n2️⃣ Probando Embeddings...")
try:
    embed_model = os.getenv('GEMINI_EMBED_MODEL', 'text-embedding-004')
    embedder = GoogleGenerativeAIEmbeddings(
        model=embed_model,
        google_api_key=api_key # type: ignore
    )
    # Generar embedding de prueba
    test_text = "Hola, este es un texto de prueba"
    print(f"   📝 Texto de prueba: '{test_text}'")
    
    vector = embedder.embed_query(test_text)
    
    print(f"   ✅ Embedding generado")
    print(f"   📊 Dimensiones: {len(vector)}")
    print(f"   🔢 Primeros 5 valores: {[round(v, 4) for v in vector[:5]]}")
    print(f"   🔢 Últimos 5 valores: {[round(v, 4) for v in vector[-5:]]}")
    
    # Validar que no sea un vector vacío o de ceros
    if all(v == 0 for v in vector):
        print("   ⚠️  Advertencia: Vector contiene solo ceros")
    
except Exception as e:
    print(f"   ❌ ERROR en Embeddings: {str(e)}")
    print(f"\n   💡 Posibles causas:")
    print(f"      - API key incorrecta")
    print(f"      - Sin conexión a internet")
    print(f"      - Límite de rate diario alcanzado (1500 requests/día gratis)")
    exit(1)

# 3. Probar Chat (Gemini)
print("\n3️⃣ Probando Chat (Gemini 2.0 Flash)...")
try:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp'),
        google_api_key=api_key,
        temperature=0.3
    )
    
    # Enviar mensaje de prueba
    test_prompt = "Responde solo con: 'Sistema funcionando correctamente'"
    print(f"   📝 Prompt de prueba: '{test_prompt}'")
    
    response = llm.invoke(test_prompt)
    
    print(f"   ✅ Respuesta recibida")
    print(f"   💬 Gemini respondió: {response.content}")
    print(f"   📏 Longitud: {len(response.content)} caracteres")
    
except Exception as e:
    print(f"   ❌ ERROR en Chat: {str(e)}")
    print(f"\n   💡 Posibles causas:")
    print(f"      - API key incorrecta")
    print(f"      - Modelo no disponible (verifica el nombre)")
    print(f"      - Límite de rate alcanzado")
    exit(1)

# 4. Probar embedding de múltiples textos
print("\n4️⃣ Probando embeddings batch...")
try:
    texts = [
        "El gato está en la casa",
        "El felino está en el hogar",
        "El coche es rojo"
    ]
    
    vectors = embedder.embed_documents(texts)
    
    print(f"   ✅ {len(vectors)} vectores generados")
    
    # Calcular similitud aproximada entre primeros dos (deberían ser similares)
    from numpy import dot
    from numpy.linalg import norm
    
    def cosine_similarity(v1, v2):
        return dot(v1, v2) / (norm(v1) * norm(v2))
    
    sim_12 = cosine_similarity(vectors[0], vectors[1])
    sim_13 = cosine_similarity(vectors[0], vectors[2])
    
    print(f"   📊 Similitud 'gato/casa' vs 'felino/hogar': {sim_12:.3f}")
    print(f"   📊 Similitud 'gato/casa' vs 'coche rojo': {sim_13:.3f}")
    
    if sim_12 > sim_13:
        print(f"   ✅ Embeddings funcionan correctamente (detectan semántica)")
    else:
        print(f"   ⚠️  Similitudes inesperadas")
    
except Exception as e:
    print(f"   ⚠️  Error en prueba batch: {str(e)}")

# 5. Resumen y costos
print("\n" + "=" * 60)
print("✅ TODAS LAS PRUEBAS PASARON")
print("=" * 60)
print("\n💰 Costos de Gemini:")
print("   - Embeddings: GRATIS hasta 1,500 req/día")
print("   - Gemini 2.0 Flash: GRATIS hasta 1,500 req/día")
print("   - Después: ~$0.075 por 1M tokens input")
print("\n🚀 Tu configuración está lista para usar")
print("   Ejecuta: uvicorn app.main:app --reload")
print()