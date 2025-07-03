import asyncio
import aiohttp
import pandas as pd
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Cargar variables de entorno desde archivo .env si existe
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuración de modelos
MODELS = {
    "deepseek": "deepseek/deepseek-chat-v3-0324",
    "gemini": "google/gemini-flash-1.5-8b",
    "claude": "anthropic/claude-3-haiku",
    "gpt4_mini": "openai/gpt-4.1-mini"
}

# Configuración para pruebas (cambiar estos valores para testing)
TEST_MODE = os.getenv('TEST_MODE', 'true').lower() == 'true'
TEST_ROWS = int(os.getenv('TEST_ROWS', '5'))  # Número de filas para pruebas
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))  # Tamaño de lote
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '20'))  # Máximo de requests concurrentes

# Configuración de API
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise ValueError("Por favor configura la variable de entorno OPENROUTER_API_KEY")

# Archivos
INPUT_FILE = os.getenv('INPUT_FILE', 'Db/SALUD_MENTAL.csv')
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'Db/SALUD_MENTAL_analyzed.csv')
CHECKPOINT_FILE = os.getenv('CHECKPOINT_FILE', 'checkpoint.json')

# Prompt template
PROMPT_TEMPLATE = """Prompt: Análisis de Sentimiento y Emoción

Eres un analizador de texto especializado en salud mental de jóvenes y evaluación de:
- Sentimiento
- Emoción predominante

**Instrucciones de formato de salida (MUY IMPORTANTE):**
Tu respuesta DEBE ser un único string, sin saltos de línea, explicaciones, o texto introductorio.
Sigue ESTRICTAMENTE el formato: `[Nivel de sentimiento con descripción]: [valor numérico], [emoción] [intensidad]`

**Ejemplos de salida CORRECTA:**
- `Positivo moderado: +7, joy 4`
- `Negativo fuerte: -8, anger 5`
- `Neutro: 0, neutro 1`
- `Negativo leve: -3, sadness 2`

**Instrucciones de análisis:**
1.  **Sentimiento:** Escala de -10 (extremadamente negativo) a +10 (extremadamente positivo). 0 es neutro.
2.  **Emoción predominante:** Elige UNA de las siguientes y asigna intensidad de 1 a 5: `neutro`, `joy`, `sadness`, `anger`, `disgust`, `surprise`, `happiness`.

**Publicación a analizar (en texto plano):**
Source: {source}
User profile: {user_profile}
Views: {views}
Shares: {shares}
Comments: {comments}
Text of the post: {text}
Language: {language}
Blue verification: {blue_verification}
Tipo de post: {tipo_post}
Si es video: {si_es_video}
Number of videos: {number_videos}
Duration in seconds: {duration}

**Tu análisis final (solo el string con el formato requerido):**
"""


class SentimentAnalyzer:
    def __init__(self):
        self.session = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.checkpoint_data = self.load_checkpoint()
        
    def load_checkpoint(self) -> Dict:
        """Cargar checkpoint si existe"""
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    data = json.load(f)
                logger.info(f"Checkpoint cargado: {data['processed_rows']} filas procesadas")
                return data
            except Exception as e:
                logger.error(f"Error cargando checkpoint: {e}")
        return {'processed_rows': 0, 'results': {}}
    
    def save_checkpoint(self, processed_rows: int, results: Dict):
        """Guardar checkpoint"""
        checkpoint = {
            'processed_rows': processed_rows,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            logger.info(f"Checkpoint guardado: {processed_rows} filas procesadas")
        except Exception as e:
            logger.error(f"Error guardando checkpoint: {e}")
    
    async def create_session(self):
        """Crear sesión aiohttp"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Cerrar sesión aiohttp"""
        if self.session:
            await self.session.close()
    
    def prepare_prompt(self, row: pd.Series) -> str:
        """Preparar prompt con los datos disponibles"""
        # Si solo tenemos la columna 'text', simular datos realistas para el análisis
        if len(row.index) == 1 and 'text' in row.index:
            # Generar followers aleatorio pero consistente basado en el hash del texto
            import hashlib
            text_hash = int(hashlib.md5(str(row.get('text', '')).encode()).hexdigest(), 16)
            
            # Distribución realista de followers
            follower_ranges = [
                (0.40, (100, 5000)),      # 40% nano influencers
                (0.30, (5000, 15000)),    # 30% nano-micro
                (0.20, (15000, 100000)),  # 20% micro
                (0.08, (100000, 500000)), # 8% micro-macro
                (0.02, (500000, 2000000)) # 2% macro-mega
            ]
            
            rand_val = (text_hash % 100) / 100
            cumulative = 0
            followers = 1000  # default
            
            for prob, (min_f, max_f) in follower_ranges:
                cumulative += prob
                if rand_val <= cumulative:
                    followers = min_f + (text_hash % (max_f - min_f))
                    break
            
            # Generar otros valores correlacionados
            views = int(followers * (2 + (text_hash % 10) / 10))  # 2x-3x followers
            shares = int(views * 0.01 * (1 + (text_hash % 5) / 10))  # 1-1.5% of views
            comments = int(views * 0.005 * (1 + (text_hash % 3) / 10))  # 0.5-0.8% of views
            
            data = {
                'source': 'TikTok',
                'user_profile': f'User_{text_hash % 10000}',
                'followers': followers,
                'views': views,
                'shares': shares,
                'comments': comments,
                'text': row.get('text', ''),
                'language': 'Spanish',
                'blue_verification': 'Yes' if followers > 100000 else 'No',
                'tipo_post': 'Video',
                'si_es_video': 'Yes',
                'number_videos': 50 + (text_hash % 500),
                'duration': 30 + (text_hash % 90)
            }
        else:
            # Usar valores reales si existen
            data = {
                'source': row.get('source', 'Unknown'),
                'user_profile': row.get('Nombre', 'Unknown'),
                'followers': row.get('Followers', 0),
                'views': row.get('views', 0),
                'shares': row.get('shares', 0),
                'comments': row.get('comments', 0),
                'text': row.get('text', ''),
                'language': row.get('Language', 'Unknown'),
                'blue_verification': row.get('Blue verification Tick', 'No'),
                'tipo_post': row.get('Tipo de post', 'Unknown'),
                'si_es_video': row.get('Si es video', 'No'),
                'number_videos': row.get('Number of videos published by author', 0),
                'duration': row.get('Duration in seconds', 0)
            }
        
        return PROMPT_TEMPLATE.format(**data)
    
    async def call_llm(self, model_name: str, model_id: str, prompt: str, row_index: int) -> Tuple[int, str, str]:
        """Llamar a un LLM específico"""
        async with self.semaphore:
            try:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/user/repo",
                    "X-Title": "Sentiment Analysis"
                }
                
                data = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "Eres un analizador de sentimientos experto."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 100
                }
                
                start_time = time.time()
                
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content'].strip()
                        elapsed = time.time() - start_time
                        logger.info(f"Fila {row_index} - {model_name}: OK ({elapsed:.2f}s)")
                        return row_index, model_name, content
                    else:
                        error_text = await response.text()
                        logger.error(f"Fila {row_index} - {model_name}: Error {response.status} - {error_text}")
                        return row_index, model_name, f"Error: {response.status}"
                        
            except asyncio.TimeoutError:
                logger.error(f"Fila {row_index} - {model_name}: Timeout")
                return row_index, model_name, "Error: Timeout"
            except Exception as e:
                logger.error(f"Fila {row_index} - {model_name}: {str(e)}")
                return row_index, model_name, f"Error: {str(e)}"
    
    async def process_row(self, row: pd.Series, row_index: int) -> Dict[str, str]:
        """Procesar una fila con todos los modelos"""
        # Verificar si ya fue procesada
        row_key = str(row_index)
        if row_key in self.checkpoint_data['results']:
            logger.info(f"Fila {row_index} ya procesada, usando resultado del checkpoint")
            return self.checkpoint_data['results'][row_key]
        
        prompt = self.prepare_prompt(row)
        
        # Llamar a todos los modelos en paralelo
        tasks = []
        for model_name, model_id in MODELS.items():
            task = self.call_llm(model_name, model_id, prompt, row_index)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Organizar resultados
        row_results = {}
        for _, model_name, result in results:
            row_results[f"result_{model_name}"] = result
        
        return row_results
    
    async def process_batch(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> List[Dict]:
        """Procesar un lote de filas"""
        batch_results = []
        
        for idx in range(start_idx, min(end_idx, len(df))):
            row = df.iloc[idx]
            results = await self.process_row(row, idx)
            batch_results.append((idx, results))
            
            # Actualizar checkpoint después de cada fila
            self.checkpoint_data['results'][str(idx)] = results
            self.save_checkpoint(idx + 1, self.checkpoint_data['results'])
        
        return batch_results
    
    async def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analizar todo el dataframe"""
        await self.create_session()
        
        try:
            # Determinar filas a procesar
            start_row = self.checkpoint_data['processed_rows']
            total_rows = len(df) if not TEST_MODE else min(TEST_ROWS, len(df))
            
            logger.info(f"Procesando desde fila {start_row} hasta {total_rows}")
            
            # Crear columnas para resultados si no existen
            for model_name in MODELS.keys():
                col_name = f"result_{model_name}"
                if col_name not in df.columns:
                    df[col_name] = ""
            
            # Procesar en lotes
            for batch_start in range(start_row, total_rows, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_rows)
                logger.info(f"Procesando lote: filas {batch_start} a {batch_end}")
                
                batch_results = await self.process_batch(df, batch_start, batch_end)
                
                # Actualizar dataframe con resultados
                for idx, results in batch_results:
                    for col_name, value in results.items():
                        df.at[idx, col_name] = value
                
                # Guardar progreso
                df.to_csv(OUTPUT_FILE, index=False)
                logger.info(f"Progreso guardado en {OUTPUT_FILE}")
                
                # Pequeña pausa entre lotes para no saturar la API
                if batch_end < total_rows:
                    await asyncio.sleep(1)
            
            logger.info("Análisis completado!")
            
            # Limpiar checkpoint si se completó todo
            if not TEST_MODE and os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                logger.info("Checkpoint eliminado")
            
            return df
            
        finally:
            await self.close_session()


def main():
    """Función principal"""
    logger.info("=== Iniciando análisis de sentimientos ===")
    logger.info(f"Modo de prueba: {TEST_MODE}")
    if TEST_MODE:
        logger.info(f"Procesando solo {TEST_ROWS} filas")
    
    # Cargar CSV
    try:
        df = pd.read_csv(INPUT_FILE)
        logger.info(f"CSV cargado: {len(df)} filas totales")
    except Exception as e:
        logger.error(f"Error cargando CSV: {e}")
        return
    
    # Crear analizador y procesar
    analyzer = SentimentAnalyzer()
    
    # Ejecutar análisis asíncrono
    df_analyzed = asyncio.run(analyzer.analyze_dataframe(df))
    
    # Guardar resultado final
    df_analyzed.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Resultados guardados en {OUTPUT_FILE}")
    
    # Mostrar estadísticas
    if TEST_MODE:
        logger.info("\n=== Muestra de resultados ===")
        for model_name in MODELS.keys():
            col_name = f"result_{model_name}"
            logger.info(f"\n{model_name}:")
            for i in range(min(3, len(df_analyzed))):
                if df_analyzed.at[i, col_name]:
                    logger.info(f"  Fila {i}: {df_analyzed.at[i, col_name][:50]}...")


if __name__ == "__main__":
    main() 