# Engram per gestione memoria LLM

Progetto per ottimizzazione della memoria degli agenti LLM tramite engrammi.

## Descrizione

Un nuovo approccio per la gestione della memoria degli agenti LLM, migliorando l'efficienza e le prestazioni attraverso la modellazione degli engrammi come unità di memoria persistente, con operazioni CRUD e indicizzazione vettoriale per ricerca rapida.

## Architettura

Il progetto consiste nei seguenti componenti principali:

- `engram_manager.py`: modulo core che implementa:
  - `EngramStore`: classe per operazioni CRUD sugli engrammi, con supporto per storage in memoria o su file JSON Lines.
  - `EngramIndex`: classe per indicizzazione e ricerca di similarità utilizzando FAISS (o alternativa basata su numpy per prototipi).
  - Funzioni di utilità per serializzazione, generazione di ID e timestamp.
- `benchmark_gpu.sh`: script di benchmark per valutare le prestazioni di inserimento, ricerca e aggiornamento su GPU.
- `ENGRAM_SCHEMA.md`: documento che definisce lo schema JSON degli engrammi.
- `tests/test_engram_manager.py`: suite di test unitari per validare il funzionamento del core.

## Installazione

1. Clonare il repository:
   ```bash
   git clone <repository-url>
   cd engram-per-gestione-memoria-llm
   ```
2. Creare un ambiente virtuale (opzionale ma consigliato):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Installare le dipendenze:
   ```bash
   pip install --user -r requirements.txt
   ```
   Le dipendenze principali sono:
   - numpy
   - faiss-cpu (o faiss-gpu se disponibile)
   - (eventuali altre richieste dai test)

## Uso

### Avvio rapido

```python
from engram_manager import EngramStore, EngramIndex
import numpy as np

# Inizializza lo store (opzionale: specificare un percorso per persistenza su file)
store = EngramStore(storage_path="./engrams.jsonl")
index = EngramIndex(dimension=128)  # dimensione dell'embedding

# Creare un engram di esempio
engram = {
    "id": "",  # verrà generato automaticamente
    "timestamp": "",  # verrà generato automaticamente se vuoto
    "embedding": np.random.rand(128).tolist(),
    "metadata": {"source": "test", "category": "example"}
}

# Aggiungere allo store
engram_id = store.add_engram(engram)
print(f"Aggiunto engram con ID: {engram_id}")

# Aggiungere all'indice
index.add_vectors([engram["embedding"]], [engram_id])

# Ricerca di similarità
query_vector = np.random.rand(128).tolist()
distances, returned_ids = index.search(query_vector, k=5)
print(f"Risultati più simili: {list(zip(returned_ids, distances))}")
```

### Persistenza

Lo store salva automaticamente su file se `storage_path` è fornito. Il formato è JSON Lines, un oggetto JSON per riga.

## Esempi

Vedere la directory `examples/` per script di utilizzo più dettagliati (da creare).

## Stato

✅ COMPLETATO — 2026-06-11
- Fase 1/5 — Inizializzazione progetto: completata
- Fase 2/5 — Definizione schema memoria: completata
- Fase 3/5 — Implementazione core manager: completata
- Fase 4/5 — Script di benchmark manuale: completata
- Fase 5/5 — Aggiornamento vault di sistema: completata

Tutti gli obiettivi del progetto sono stati raggiunti. Il codice è pronto per l'uso e l'integrazione in sistemi LLM locali.