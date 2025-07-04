#!/usr/bin/env python3
"""
Script de prueba para verificar el análisis de sentimientos en un directorio.
"""
import os
import sys
import pandas as pd
from pathlib import Path

# Cargar variables de entorno desde archivo .env si existe
try:
    from dotenv import load_dotenv
    if load_dotenv():
        print("✅ Archivo .env cargado")
except ImportError:
    pass

# --- Configuración del entorno de prueba ---
os.environ['MAX_ROWS_TO_PROCESS'] = '5'
os.environ['BATCH_SIZE'] = '3'
os.environ['MAX_CONCURRENT_REQUESTS'] = '5'
os.environ['INPUT_DIR'] = 'Db/'
os.environ['OUTPUT_FILE'] = 'Db/all_comments_analyzed_test.csv'
os.environ['CHECKPOINT_FILE'] = 'checkpoint_all_test.json'

# --- Nuevas variables para seleccionar archivos y modelo ---
os.environ['FILES_TO_PROCESS'] = 'AI COMMMENTS.csv,AI INFLU.csv'
os.environ['MODEL_ID'] = 'google/gemini-flash-1.5'
os.environ['MODEL_NAME'] = 'gemini'

# --- Verificaciones iniciales ---

# 1. Verificar API key
if not os.getenv('OPENROUTER_API_KEY'):
    print("❌ ERROR: No se encontró OPENROUTER_API_KEY")
    print("\nPor favor configura tu API key de una de estas formas:")
    print("1. Crear archivo .env con: OPENROUTER_API_KEY=tu_key_aqui")
    print("2. Exportar variable: export OPENROUTER_API_KEY=tu_key_aqui")
    sys.exit(1)
print("✅ API Key configurada")

# 2. Verificar directorio de entrada
input_dir = Path(os.environ['INPUT_DIR'])
if not input_dir.is_dir():
    print(f"❌ ERROR: No se encontró el directorio de entrada {input_dir}")
    sys.exit(1)
print(f"✅ Directorio de entrada encontrado: {input_dir}")

# 3. Limpiar checkpoint anterior si existe
checkpoint_file = Path(os.environ['CHECKPOINT_FILE'])
if checkpoint_file.exists():
    print(f"🗑️ Borrando checkpoint de prueba anterior: '{checkpoint_file}'")
    checkpoint_file.unlink()

# --- Ejecución de la prueba ---

print("\n" + "="*50)
print("🚀 INICIANDO PRUEBA DE ANÁLISIS DE DIRECTORIO 🚀")
print("="*50)
print("\n⚙️ Configuración:")
print(f"   - Directorio de entrada: {os.getenv('INPUT_DIR')}")
print(f"   - Archivo de salida:     {os.getenv('OUTPUT_FILE')}")
print(f"   - Filas a procesar:      {os.getenv('MAX_ROWS_TO_PROCESS')}")
print(f"   - Archivos de prueba:    {os.getenv('FILES_TO_PROCESS')}")
print(f"   - Modelo:                {os.getenv('MODEL_ID')}")
print(f"   - Procesamiento concurrente: Sí")

# Importar y ejecutar el script principal
try:
    from analyze_directory import main
    print("\n🔄 Iniciando análisis de prueba...\n")
    main()
    
    # --- Verificación de resultados ---
    output_file = Path(os.environ['OUTPUT_FILE'])
    if output_file.exists():
        df_result = pd.read_csv(output_file)
        print("\n✅ ¡Análisis de prueba completado!")
        print(f"\n📊 Resultados guardados en: {output_file}")
        
        print(f"\n📝 Mostrando las primeras {min(len(df_result), 5)} filas del resultado:")
        print(df_result.head())

        # Mostrar muestra detallada de resultados
        print("\n🔍 Muestra detallada de los resultados del análisis:")
        result_cols = [col for col in df_result.columns if col.startswith(os.environ['MODEL_NAME'] + '_')]
        
        rows_to_show = min(int(os.environ['MAX_ROWS_TO_PROCESS']), len(df_result))
        
        for i in range(rows_to_show):
            # Verificar si la fila tiene algún dato de resultado
            if not df_result.iloc[i][result_cols].isnull().all():
                print(f"\n--- Fila {i} ---")
                print(f"Texto original: {df_result.iloc[i]['text'][:100]}...")
                print(f"Archivo fuente: {df_result.iloc[i]['source_file']}")
                for col in result_cols:
                    if pd.notna(df_result.iloc[i][col]):
                        print(f"  - {col}: {df_result.iloc[i][col]}")
            else:
                print(f"\n--- Fila {i} ---")
                print("  - Sin resultados de análisis (puede haber sido saltada o fallado).")
                
    else:
        print(f"❌ ERROR: No se encontró el archivo de resultados esperado en '{output_file}'")
        
except Exception as e:
    print(f"\n❌ ERROR catastrófico durante la ejecución de la prueba: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Limpieza opcional de archivos de prueba
    output_file_path = Path(os.getenv('OUTPUT_FILE'))
    checkpoint_file_path = Path(os.getenv('CHECKPOINT_FILE'))
    
    if output_file_path.exists():
        print(f"\n🧹 Limpiando archivo de salida de prueba: {output_file_path}")
        # output_file_path.unlink() # Descomentar para borrar automáticamente
    
    if checkpoint_file_path.exists():
        print(f"🧹 Limpiando archivo de checkpoint de prueba: {checkpoint_file_path}")
        # checkpoint_file_path.unlink() # Descomentar para borrar automáticamente

    print("\n🏁 Prueba finalizada.") 