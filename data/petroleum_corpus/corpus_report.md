# Petroleum RAG Corpus Report

Generated: 2026-05-19T19:14:53.544435+00:00

## Summary

- Documents: 12
- Chunks: 116
- Total words: 33304
- Mean document words: 2775.3
- Mean chunk words: 334.5

## Documents

| Topic | Title | Words | URL |
|---|---:|---:|---|
| oil_basics | Oil and petroleum products explained - U.S. Energy Information Administration (EIA) | 725 | https://www.eia.gov/energyexplained/oil-and-petroleum-products/ |
| oil_supply_and_production | Where our oil comes from in depth - U.S. Energy Information Administration (EIA) | 1173 | https://www.eia.gov/energyexplained/oil-and-petroleum-products/where-our-oil-comes-from-in-depth.php |
| refining | Refining crude oil - the refining process - U.S. Energy Information Administration (EIA) | 910 | https://www.eia.gov/energyexplained/oil-and-petroleum-products/refining-crude-oil-the-refining-process.php |
| natural_gas_basics | Natural gas explained - U.S. Energy Information Administration (EIA) | 1411 | https://www.eia.gov/energyexplained/natural-gas/ |
| petroleum_systems | Petroleum Systems and Geologic Assessment of Oil and Gas in the San Joaquin Basin Province, California | 1327 | https://pubs.usgs.gov/pp/pp1713/ |
| petroleum_systems_pdf | USGS Chapter PS: Petroleum Systems | 25611 | https://pubs.usgs.gov/of/1998/ofr-98-0034/PS.pdf |
| permeability | permeability \| Energy Glossary | 476 | https://glossary.slb.com/en/terms/p/permeability |
| porosity | effective porosity \| Energy Glossary | 362 | https://glossary.slb.com/en/terms/e/effective_porosity |
| pore_pressure | pore pressure \| Energy Glossary | 262 | https://glossary.slb.com/en/terms/p/pore_pressure |
| well_logging | log \| Energy Glossary | 661 | https://glossary.slb.com/en/terms/l/log |
| drilling_fluids | drilling fluid \| Energy Glossary | 226 | https://glossary.slb.com/en/terms/d/drilling_fluid |
| artificial_lift | gas lift \| Energy Glossary | 160 | https://glossary.slb.com/en/terms/g/gas_lift |

## Topic Coverage

- artificial_lift: 1 document(s)
- drilling_fluids: 1 document(s)
- natural_gas_basics: 1 document(s)
- oil_basics: 1 document(s)
- oil_supply_and_production: 1 document(s)
- permeability: 1 document(s)
- petroleum_systems: 1 document(s)
- petroleum_systems_pdf: 1 document(s)
- pore_pressure: 1 document(s)
- porosity: 1 document(s)
- refining: 1 document(s)
- well_logging: 1 document(s)

## Suggested Notebook Diagnostics

- Corpus statistics table: documents, chunks, mean/median chunk length.
- Retrieval check: for each test question, show top-3 retrieved chunks with source URL.
- Metric: recall@k using `expected_answer_keywords` from `rag_test_questions.json`.
- Heatmap: cosine similarity between each test question embedding and each document or chunk embedding.
