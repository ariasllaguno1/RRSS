#!/usr/bin/env python3
"""
Script de prueba para verificar el análisis de sentimientos
"""
import os
import sys
import asyncio
import pandas as pd
from pathlib import Path

# Cargar variables de entorno desde archivo .env si existe
try:
    from dotenv import load_dotenv
    if load_dotenv():
        print("✅ Archivo .env cargado")
except ImportError:
    pass

# Configurar variables de entorno para prueba
os.environ['TEST_MODE'] = 'true'
os.environ['TEST_ROWS'] = '3'
os.environ['BATCH_SIZE'] = '2'
os.environ['MAX_CONCURRENT_REQUESTS'] = '4'

# Verificar API key
if not os.getenv('OPENROUTER_API_KEY'):
    print("❌ ERROR: No se encontró OPENROUTER_API_KEY")
    print("\nPor favor configura tu API key de una de estas formas:")
    print("1. Crear archivo .env con: OPENROUTER_API_KEY=tu_key_aqui")
    print("2. Exportar variable: export OPENROUTER_API_KEY=tu_key_aqui")
    print("3. En Windows: set OPENROUTER_API_KEY=tu_key_aqui")
    sys.exit(1)

print("✅ API Key configurada")

# Verificar archivo de entrada
input_file = Path("Db/SALUD_MENTAL.csv")
if not input_file.exists():
    print(f"❌ ERROR: No se encontró el archivo {input_file}")
    sys.exit(1)

# Mostrar información del archivo
df = pd.read_csv(input_file)
print(f"\n📊 Archivo CSV cargado:")
print(f"   - Total de filas: {len(df)}")
print(f"   - Columnas: {list(df.columns)}")
print(f"\n📝 Primeras 3 filas:")
print(df.head(3))

# Preguntar si continuar
print("\n" + "="*50)

# Opción para borrar checkpoint
checkpoint_file = Path(os.getenv('CHECKPOINT_FILE', 'checkpoint.json'))
if checkpoint_file.exists():
    reset_response = input(f"Se encontró un checkpoint ('{checkpoint_file}'). ¿Quieres borrarlo y empezar de nuevo? (s/n): ")
    if reset_response.lower() == 's':
        checkpoint_file.unlink()
        print("✅ Checkpoint borrado.")

print("\n🚀 Configuración de prueba:")
print(f"   - Filas a procesar: {os.getenv('TEST_ROWS')}")
print(f"   - Modelos: DeepSeek, Gemini, Claude, GPT-4.1 Mini")
print(f"   - Procesamiento en paralelo: Sí")
print(f"   - Especialista en salud mental de jóvenes")
print("="*50)

response = input("\n¿Deseas ejecutar la prueba? (s/n): ")
if response.lower() != 's':
    print("Prueba cancelada.")
    sys.exit(0)

# Importar y ejecutar el script principal
try:
    from analyze_sentiment_llms import main
    print("\n🔄 Iniciando análisis de prueba...\n")
    main()
    
    # Verificar resultados
    output_file = Path("Db/SALUD_MENTAL_analyzed.csv")
    if output_file.exists():
        df_result = pd.read_csv(output_file)
        print("\n✅ Análisis completado!")
        print(f"\n📊 Resultados guardados en: {output_file}")
        
        # Mostrar muestra de resultados
        print("\n📝 Muestra de resultados:")
        result_cols = [col for col in df_result.columns if col.startswith('result_')]
        
        for i in range(min(3, len(df_result))):
            if any(df_result.iloc[i][col] for col in result_cols if col in df_result.columns):
                print(f"\nFila {i}:")
                print(f"Texto: {df_result.iloc[i]['text'][:100]}...")
                for col in result_cols:
                    if col in df_result.columns and df_result.iloc[i][col]:
                        print(f"{col}: {df_result.iloc[i][col]}")
    else:
        print("❌ No se encontró archivo de resultados")
        
except Exception as e:
    print(f"\n❌ Error durante la ejecución: {e}")
    import traceback
    traceback.print_exc() 