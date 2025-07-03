# Script de Análisis de Sentimientos con Múltiples LLMs

Este script analiza sentimientos, emociones y tamaño de influencer usando múltiples modelos de lenguaje a través de OpenRouter.

## Características

- ✅ Procesamiento paralelo de múltiples LLMs
- ✅ Sistema de checkpoints para reanudar procesamiento
- ✅ Modo de prueba configurable
- ✅ Logs detallados
- ✅ Compatible con GitHub Actions

## Modelos utilizados

1. **DeepSeek**: `deepseek/deepseek-chat-v3-0324`
2. **Google Gemini**: `google/gemini-flash-1.5-8b`
3. **Anthropic Claude**: `anthropic/claude-3-haiku`
4. **OpenAI GPT-4 Nano**: `openai/gpt-4-1-nano`

## Configuración

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

Crea un archivo `.env` con las siguientes variables:

```env
# API Key de OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Configuración de pruebas
TEST_MODE=true  # Cambiar a false para procesar todo el CSV
TEST_ROWS=5     # Número de filas para procesar en modo de prueba

# Configuración de procesamiento
BATCH_SIZE=10                  # Tamaño de lote para procesamiento
MAX_CONCURRENT_REQUESTS=20     # Máximo de requests concurrentes a la API

# Archivos
INPUT_FILE=Db/SALUD_MENTAL.csv
OUTPUT_FILE=Db/SALUD_MENTAL_analyzed.csv
CHECKPOINT_FILE=checkpoint.json
```

## Uso

### Modo de prueba (por defecto)

```bash
python analyze_sentiment_llms.py
```

Esto procesará solo 5 filas del CSV para probar que todo funciona correctamente.

### Modo completo

Configura `TEST_MODE=false` en tu archivo `.env` o variable de entorno:

```bash
TEST_MODE=false python analyze_sentiment_llms.py
```

### Reanudar desde checkpoint

Si el script se interrumpe, simplemente vuelve a ejecutarlo. Automáticamente:
- Detectará el checkpoint existente
- Continuará desde la última fila procesada
- No repetirá análisis ya completados

## Salida

El script genera:

1. **CSV con resultados**: `Db/SALUD_MENTAL_analyzed.csv`
   - Columnas originales del CSV
   - `result_deepseek`: Resultado del modelo DeepSeek
   - `result_gemini`: Resultado del modelo Gemini
   - `result_claude`: Resultado del modelo Claude
   - `result_gpt4_nano`: Resultado del modelo GPT-4 Nano

2. **Archivo de log**: `analysis.log`
   - Progreso detallado
   - Tiempos de respuesta
   - Errores si los hay

3. **Checkpoint**: `checkpoint.json` (se elimina al completar)

## Formato de salida

Cada modelo devuelve un string con el formato:
```
[Nivel de sentimiento con descripción]: [valor numérico], [emoción] [intensidad], [tamaño influencer]
```

Ejemplo:
```
Positivo moderado: +7, joy 4, nano
```

## GitHub Actions

Para usar en GitHub Actions, agrega el siguiente workflow:

```yaml
name: Análisis de Sentimientos

on:
  workflow_dispatch:
  push:
    branches: [ main ]

jobs:
  analyze:
    runs-on: ubuntu-latest
    timeout-minutes: 360  # 6 horas máximo
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run analysis
      env:
        OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        TEST_MODE: false
        BATCH_SIZE: 20
        MAX_CONCURRENT_REQUESTS: 30
      run: |
        python analyze_sentiment_llms.py
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: analysis-results
        path: |
          Db/SALUD_MENTAL_analyzed.csv
          analysis.log
```

## Optimización

Para acelerar el procesamiento:

1. **Aumentar concurrencia**: `MAX_CONCURRENT_REQUESTS=50`
2. **Aumentar tamaño de lote**: `BATCH_SIZE=50`
3. **Usar menos modelos**: Comentar modelos en el diccionario `MODELS`

## Manejo de errores

El script maneja automáticamente:
- Timeouts de API
- Errores de red
- Límites de rate de la API
- Interrupciones del proceso

Los errores se registran pero no detienen el procesamiento de otras filas. 