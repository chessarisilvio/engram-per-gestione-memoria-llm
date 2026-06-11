# Schema JSON per la memorizzazione degli engrammi

Questo schema definisce la struttura dei dati per la memorizzazione degli engrammi nel sistema di gestione della memoria LLM.

## Struttura dell'engramma

Ogni engramma è rappresentato come un oggetto JSON con i seguenti campi:

```json
{
  "id": "string (UUID v4)",
  "timestamp": "string (ISO 8601 format)",
  "embedding": "array of numbers (float32)",
  "metadata": {
    "source": "string (optional)",
    "tags": "array of strings (optional)",
    "score": "number (float, optional)",
    "custom": "object (optional)"
  }
}
```

### Descrizione dei campi

- **id**: Identificatore unico dell'engramma. Si consiglia l'uso di UUID v4 per garantire l'unicità.
- **timestamp**: Data e ora di creazione dell'engramma in formato ISO 8601 (es. "2026-06-11T21:22:00Z").
- **embedding**: Vettore di embedding rappresentante l'engramma. È un array di numeri a virgola mobile (float32). La dimensione dipende dal modello di embedding utilizzato.
- **metadata**: Oggetto opzionale per informazioni aggiuntive:
  - **source**: Stringa che indica l'origine dell'engramma (es. "user_input", "system_generated").
  - **tags**: Array di stringhe per categorizzare l'engramma.
  - **score**: Punteggio di rilevanza o importanza (es. da 0.0 a 1.0).
  - **custom**: Oggetto libero per campi specifici dell'applicazione.

### Esempio di engramma

```json
{
  "id": "a1b2c3d4-e5f6-7890-g1h2-i3j4k5l6m7n8",
  "timestamp": "2026-06-11T21:22:00Z",
  "embedding": [0.123, -0.456, 0.789, ...],
  "metadata": {
    "source": "user_input",
    "tags": ["conversation", "topic:ai"],
    "score": 0.85,
    "custom": {
      "session_id": "sess_abc123",
      "model_used": "qwen3.6-35b"
    }
  }
}
```

### Note sull'implementazione

- L'embedding può essere memorizzato come array di float32 per efficienza. In fase di serializzazione/deserializzazione, assicurarsi di mantenere la precisione necessaria.
- Il campo `metadata` è progettato per essere estensibile senza modificare lo schema di base.
- Per la ricerca di similarità, l'embedding verrà confrontato usando metriche di distanza (es. cosine similarity, L2 norm).