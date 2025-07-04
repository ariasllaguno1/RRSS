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
import hashlib
import re

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
        logging.FileHandler('analysis_directory.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuración de modelo (ahora configurable por variables de entorno)
DEFAULT_MODEL_ID = "google/gemini-flash-1.5"
DEFAULT_MODEL_NAME = "gemini"

MODEL_ID = os.getenv('MODEL_ID', DEFAULT_MODEL_ID)
MODEL_NAME = os.getenv('MODEL_NAME', DEFAULT_MODEL_NAME)

MODELS = {
    MODEL_NAME: MODEL_ID,
}

# Configuración de procesamiento (reemplaza TEST_MODE)
MAX_ROWS_TO_PROCESS = int(os.getenv('MAX_ROWS_TO_PROCESS', '-1')) # -1 para procesar todo
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '20'))

# Configuración de API
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise ValueError("Por favor configura la variable de entorno OPENROUTER_API_KEY")

# Archivos
INPUT_DIR = os.getenv('INPUT_DIR', 'Db/')
OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'Db/all_comments_analyzed.csv')
CHECKPOINT_FILE = os.getenv('CHECKPOINT_FILE', 'checkpoint_all.json')
FILES_TO_PROCESS_STR = os.getenv('FILES_TO_PROCESS', '') # Variable para especificar archivos

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
        self.checkpoint_lock = asyncio.Lock()  # Lock para escritura segura
        self.checkpoint_data = self.load_checkpoint()

    def load_checkpoint(self) -> Dict:
        """Cargar checkpoint si existe"""
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    # Prevenir error si el archivo está vacío
                    content = f.read()
                    if not content:
                        logger.warning("El archivo de checkpoint está vacío. Empezando de cero.")
                        return {'processed_rows': 0, 'results': {}}
                    data = json.loads(content)
                logger.info(f"Checkpoint cargado: {data.get('processed_rows', 0)} filas procesadas")
                # Asegurar que las claves requeridas existan
                if 'processed_rows' not in data or 'results' not in data:
                    logger.warning("Archivo de checkpoint incompleto. Recreando.")
                    return {'processed_rows': 0, 'results': {}}
                return data
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Error cargando checkpoint, se empezará de nuevo: {e}")
        return {'processed_rows': 0, 'results': {}}

    async def save_checkpoint(self, processed_rows: int, results: Dict):
        """Guardar checkpoint de forma segura, evitando retrocesos."""
        async with self.checkpoint_lock:
            current_progress = 0
            if os.path.exists(CHECKPOINT_FILE):
                try:
                    with open(CHECKPOINT_FILE, 'r') as f:
                        content = f.read()
                        if content:
                           current_progress = json.loads(content).get('processed_rows', 0)
                except (json.JSONDecodeError, FileNotFoundError):
                    # Si hay error o no existe, se considera 0
                    pass
            
            # Solo guardar si el progreso es mayor
            if processed_rows > current_progress:
                checkpoint = {
                    'processed_rows': processed_rows,
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }
                try:
                    # Escritura atómica (temporal y luego renombrar)
                    temp_file = CHECKPOINT_FILE + ".tmp"
                    with open(temp_file, 'w') as f:
                        json.dump(checkpoint, f, indent=2)
                    os.replace(temp_file, CHECKPOINT_FILE)
                    logger.info(f"Checkpoint guardado: {processed_rows} filas procesadas")
                except Exception as e:
                    logger.error(f"Error guardando checkpoint: {e}")
            else:
                logger.info(f"Se omite guardado de checkpoint. Progreso actual ({current_progress}) es mayor o igual que el nuevo ({processed_rows}).")

    async def create_session(self):
        """Crear sesión aiohttp"""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        """Cerrar sesión aiohttp"""
        if self.session:
            await self.session.close()

    def prepare_prompt(self, row: pd.Series) -> str:
        """Prepara el prompt generando metadatos realistas a partir del texto."""
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
            'source': 'Social Media',
            'user_profile': f'User_{text_hash % 10000}',
            'followers': followers,
            'views': views,
            'shares': shares,
            'comments': comments,
            'text': row.get('text', ''),
            'language': 'Unknown',
            'blue_verification': 'Yes' if followers > 100000 else 'No',
            'tipo_post': 'Post',
            'si_es_video': 'No',
            'number_videos': 50 + (text_hash % 500),
            'duration': 0
        }
        return PROMPT_TEMPLATE.format(**data)

    def parse_llm_response(self, response: str) -> Dict[str, Optional[str]]:
        """
        Parsea la respuesta del LLM para extraer sentimiento y emoción.
        Formato esperado: `[Descripción]: [valor], [emoción] [intensidad]`
        Ej: `Positivo moderado: +7, joy 4`
        """
        if response.startswith("Error:"):
            return {
                "sentiment_desc": response,
                "sentiment_score": None,
                "emotion": None,
                "emotion_intensity": None
            }

        # Patrón regex más flexible
        pattern = re.compile(
            r"^(.*?):\s*([+-]?\d+),\s*(\w+)\s+(\d+)$",
            re.IGNORECASE
        )
        match = pattern.match(response.strip())

        if match:
            return {
                "sentiment_desc": match.group(1).strip(),
                "sentiment_score": match.group(2),
                "emotion": match.group(3).lower(),
                "emotion_intensity": match.group(4)
            }
        else:
            logger.warning(f"No se pudo parsear la respuesta: '{response}'")
            return {
                "sentiment_desc": response,
                "sentiment_score": None,
                "emotion": None,
                "emotion_intensity": None
            }

    async def call_llm(self, model_name: str, model_id: str, prompt: str, row_index: int) -> Tuple[int, str, str]:
        """Llamar a un LLM específico"""
        async with self.semaphore:
            try:
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 100
                }
                start_time = time.time()
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=60)
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
        row_key = str(row_index)
        if row_key in self.checkpoint_data['results']:
            logger.info(f"Fila {row_index} ya procesada, usando resultado del checkpoint")
            return self.checkpoint_data['results'][row_key]
        
        prompt = self.prepare_prompt(row)
        
        tasks = [self.call_llm(name, id, prompt, row_index) for name, id in MODELS.items()]
        results = await asyncio.gather(*tasks)
        
        row_results = {}
        for _, model_name, result in results:
            parsed_result = self.parse_llm_response(result)
            for key, value in parsed_result.items():
                row_results[f"{model_name}_{key}"] = value
        
        return row_results

    async def process_batch(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> List[Tuple[int, Dict]] :
        """Procesar un lote de filas"""
        batch_results = []
        for idx in range(start_idx, min(end_idx, len(df))):
            row = df.iloc[idx]
            results = await self.process_row(row, idx)
            batch_results.append((idx, results))
            
            self.checkpoint_data['results'][str(idx)] = results
            # Ahora la llamada es asíncrona y más segura
            await self.save_checkpoint(idx + 1, self.checkpoint_data['results'])
        return batch_results

    async def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analizar todo el dataframe"""
        await self.create_session()
        try:
            start_row = self.checkpoint_data['processed_rows']
            
            total_rows_in_df = len(df)
            rows_to_process = total_rows_in_df
            if MAX_ROWS_TO_PROCESS != -1:
                rows_to_process = min(total_rows_in_df, MAX_ROWS_TO_PROCESS)

            logger.info(f"Total de filas en el dataframe: {total_rows_in_df}")
            logger.info(f"Filas a procesar según configuración: {rows_to_process}")
            logger.info(f"Filas ya procesadas según checkpoint: {start_row}")
            
            # Verificar si ya se completó el procesamiento
            if start_row >= rows_to_process:
                logger.info(f"El análisis ya está completo. Se procesaron {start_row} filas de {rows_to_process} requeridas.")
                
                # Cargar resultados del checkpoint al dataframe
                for model_name in MODELS.keys():
                    for suffix in ["sentiment_desc", "sentiment_score", "emotion", "emotion_intensity"]:
                        col_name = f"{model_name}_{suffix}"
                        if col_name not in df.columns:
                            df[col_name] = pd.Series(dtype='object')
                
                # Aplicar los resultados guardados al dataframe actual
                for idx_str, results in self.checkpoint_data['results'].items():
                    idx = int(idx_str)
                    if idx < len(df):  # Solo aplicar si el índice existe en el df actual
                        for col_name, value in results.items():
                            if col_name in df.columns:
                                df.at[idx, col_name] = value
                
                logger.info(f"Guardando resultados existentes en {OUTPUT_FILE}")
                df.to_csv(OUTPUT_FILE, index=False)
                return df
            
            logger.info(f"Procesando desde fila {start_row} hasta {rows_to_process}")
            
            for model_name in MODELS.keys():
                for suffix in ["sentiment_desc", "sentiment_score", "emotion", "emotion_intensity"]:
                    col_name = f"{model_name}_{suffix}"
                    if col_name not in df.columns:
                        df[col_name] = pd.Series(dtype='object')
            
            batch_tasks = []
            for batch_start in range(start_row, rows_to_process, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, rows_to_process)
                task = self.process_batch(df, batch_start, batch_end)
                batch_tasks.append(task)
            
            all_batches_results = await asyncio.gather(*batch_tasks)
            all_results = [item for batch in all_batches_results for item in batch]

            logger.info("Actualizando el dataframe con todos los resultados...")
            for idx, results in all_results:
                for col_name, value in results.items():
                    df.at[idx, col_name] = value

            logger.info(f"Guardando todos los resultados en {OUTPUT_FILE}")
            df.to_csv(OUTPUT_FILE, index=False)
            
            return df
        finally:
            await self.close_session()

def load_and_prepare_data(input_dir: str, files_to_process_str: str) -> Optional[pd.DataFrame]:
    """Carga y prepara los datos de múltiples archivos CSV."""
    dir_path = Path(input_dir)
    if not dir_path.is_dir():
        logger.error(f"El directorio de entrada no existe: {input_dir}")
        return None

    if files_to_process_str:
        # Procesar solo los archivos especificados en la cadena
        filenames = [f.strip() for f in files_to_process_str.split(',') if f.strip()]
        csv_files = []
        for f in filenames:
            file_path = dir_path / f
            if file_path.exists():
                csv_files.append(file_path)
            else:
                logger.warning(f"Archivo especificado '{f}' no encontrado en '{dir_path}'. Será ignorado.")
    else:
        # Procesar todos los archivos CSV del directorio si no se especifica ninguno
        csv_files = [f for f in dir_path.glob('*.csv') if '_analyzed' not in f.name and '_test' not in f.name]

    if not csv_files:
        logger.error("No se encontraron archivos CSV para procesar.")
        return None

    logger.info(f"Archivos CSV encontrados para procesar: {[f.name for f in csv_files]}")
    
    all_dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, on_bad_lines='warn')
            if 'text' in df.columns:
                # Mantener solo columnas esenciales y añadir fuente
                df_filtered = df[['text']].copy()
                df_filtered['source_file'] = file_path.name
                all_dfs.append(df_filtered)
            else:
                logger.warning(f"Archivo '{file_path.name}' omitido por no tener columna 'text'.")
        except Exception as e:
            logger.error(f"Error cargando el archivo {file_path.name}: {e}")

    if not all_dfs:
        logger.error("No se encontraron DataFrames válidos para procesar.")
        return None

    consolidated_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total de filas consolidadas: {len(consolidated_df)}")
    return consolidated_df

def main():
    """Función principal"""
    logger.info("=== Iniciando análisis de sentimientos en directorio ===")
    
    if MAX_ROWS_TO_PROCESS != -1:
        logger.info(f"MODO LIMITADO: Se procesará un máximo de {MAX_ROWS_TO_PROCESS} filas.")
    else:
        logger.info("MODO COMPLETO: Se procesarán todas las filas encontradas.")

    logger.info(f"Usando el modelo: {MODEL_NAME} ({MODEL_ID})")
    
    df = load_and_prepare_data(INPUT_DIR, FILES_TO_PROCESS_STR)
    if df is None or df.empty:
        logger.info("No hay datos para procesar. Finalizando.")
        return

    analyzer = SentimentAnalyzer()
    df_analyzed = asyncio.run(analyzer.analyze_dataframe(df))
    
    logger.info(f"Resultados finales guardados en {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 