# Sensitive Data Filtering Guide

## Background

As part of making pre-trained models safe and reliable, automated techniques
are used to filter out personal information and other sensitive data from
training sets. Google applied this to Gemma across billions of web documents.
The same principle applies at any scale — including domain-specific SLMs
trained on curated datasets.

## Risk Level for This Pipeline

Current training data is synthetically generated mortgage product data and
carries low inherent PII risk. Filtering becomes critical if any of the
following are introduced:

- Real customer mortgage enquiries or chat logs
- Mortgage application data
- Broker notes or case files
- User feedback collected from a deployed model

## What to Filter in UK Mortgage Data

| Category | Examples |
|---|---|
| Identity | Full names, dates of birth |
| Contact | Email addresses, phone numbers, postal addresses |
| Location | UK postcodes |
| Financial | Sort codes, account numbers, IBAN |
| Government ID | National Insurance numbers |
| Sensitive categories | Health conditions (relevant to mortgage protection), income tied to individuals |

## Library Options

| Library | Type | UK PII support | Cost |
|---|---|---|---|
| **Microsoft Presidio** | Local Python library | Yes — postcode, NI number, sort code, phone | Free |
| **scrubadub** | Local Python library | Good — built with UK data in mind | Free |
| **spaCy + custom NER** | Local Python library | Customisable | Free |
| **AWS Comprehend** | External API | Yes | Paid per call |
| **Google DLP API** | External API | Yes | Paid per call |

## Recommended Approach: Microsoft Presidio (Local)

Presidio is the strongest free option and runs entirely locally — no data
leaves the machine, no API key required, no per-call cost.

### Why local matters

- Training data must not be sent to third-party APIs, especially if it
  contains real customer information
- Sending data externally creates additional GDPR data processor obligations
  that compound EU AI Act Article 10 compliance
- No network dependency during training runs

### Installation

```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

The spaCy model is downloaded once and runs locally after that.

### Basic Usage

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def scrub_pii(text: str) -> str:
    results = analyzer.analyze(text, language="en")
    return anonymizer.anonymize(text, results).text
```

Detected entities are replaced with typed placeholders:

```
Input:  "My name is John Smith, NI number AB123456C"
Output: "My name is <PERSON>, NI number <UK_NHS>"
```

## Where in the Pipeline

The filter sits between quality filtering and training — after data has been
validated for quality but before the model ever sees it:

```
data generation → quality_filter → PII scrub → boundary mix → train
```

For the existing pipeline this maps to:

```
generate_data.py → quality_filter (data_generation.yaml) → scrub step → train_slm.py
```

Practically, the scrub step can be applied either as a post-processing pass
over the JSONL files before training, or inline inside `prepare_dataset()` in
`train_slm.py` before tokenisation.

## Connection to EU AI Act

Article 10 of the EU AI Act (Data and Data Governance) requires that training
data for high-risk AI systems:

- Is free from errors and complete
- Has appropriate data governance practices
- Addresses possible biases

Documenting that a PII scrubbing step was applied — which library, which
entity types, which fields — directly satisfies the data governance and
documentation obligations under Articles 10 and 11. It also supports the
broader GDPR accountability principle that personal data is not processed
beyond its original purpose.
