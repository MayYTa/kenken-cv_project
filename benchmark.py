import time
import statistics
import sys
import json
from pathlib import Path

# Como ejecutaste el script desde la raíz, esta es tu carpeta base
directorio_raiz = Path(__file__).resolve().parent
sys.path.append(str(directorio_raiz))

try:
    from solver import solve_detected_json
except ImportError:
    from solver.solver import solve_detected_json

def benchmark_masivo():
    tamanos = [3, 4, 5, 6, 7, 8, 9]
    iteraciones = 50 # Ejecutará 50 veces cada tablero
    
    print("\nIniciando Benchmark de CP-SAT (Escalabilidad de 3x3 a 9x9)...")
    print(f"{'Tablero':<8} | {'Promedio (ms)':<15} | {'Desv. Est. (ms)':<15} | {'Mínimo (ms)':<13} | {'Máximo (ms)':<13}")
    print("-" * 75)
    
    for n in tamanos:
        archivo_prueba = directorio_raiz / 'dataset' / 'gt' / f'{n}x{n}' / f'json{n}x{n}.json'
        
        if not archivo_prueba.exists():
            print(f"{n}x{n:<6} | Archivo no encontrado: {archivo_prueba.name}")
            continue
            
        # LEER EL JSON PARA EXTRAER EL NOMBRE DE LA PRIMERA IMAGEN
        with open(archivo_prueba, 'r', encoding='utf-8') as f:
            datos_json = json.load(f)
            # Extraemos el filename del primer elemento de la lista "images"
            nombre_tablero = datos_json['images'][0]['filename']
            
        tiempos = []
        for _ in range(iteraciones):
            inicio = time.perf_counter()
            # Pasamos el nombre extraído al parámetro que exige la función
            solucion = solve_detected_json(str(archivo_prueba), filename=nombre_tablero) 
            fin = time.perf_counter()
            tiempos.append((fin - inicio) * 1000)
            
        promedio = statistics.mean(tiempos)
        desviacion = statistics.stdev(tiempos) if len(tiempos) > 1 else 0
        minimo = min(tiempos)
        maximo = max(tiempos)
        
        print(f"{n}x{n:<6} | {promedio:<15.2f} | ± {desviacion:<13.2f} | {minimo:<13.2f} | {maximo:<13.2f}")

if __name__ == "__main__":
    benchmark_masivo()