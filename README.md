# Sistemas de recomendación sobre Amazon Reviews 2023

Proyecto de la materia **Minería de Big Data (ST1803)**. Autor: Kamma Dyhr.

El proyecto predice la calificación (1–5 estrellas) que un usuario dará a un producto de la categoría
*Beauty & Personal Care* de Amazon Reviews 2023. Todo el flujo corre en **Databricks** con **PySpark**,
**tablas Delta** y **Spark MLlib**.

## Resultados principales

| Conjunto | Modelo | RMSE validación | RMSE prueba |
| --- | --- | --- | --- |
| 5-core | Baseline de sesgos (λ_i = 10, λ_u = 5) | 1,24539 | 1,1547 |
| 5-core | Baseline + prior de texto (λ_i = 20) | **1,2424** | **1,1522** |
| Completo | Baseline de sesgos | 1,46931 | 1,43086 |
| Completo | Baseline + prior de texto (λ_i = 10) | **1,46425** | **1,42580** |

- ALS sobre residuales, filtrado colaborativo ítem–ítem y un blend por regresión ridge **no mejoran** el baseline
  (RMSE de prueba 5-core: 1,1565, 1,1562 y 1,1572).
- Un modelo TF-IDF sobre los metadatos del producto, usado como **prior del sesgo del ítem**, da una mejora pequeña
  pero consistente (0,22 % en 5-core y 0,35 % en el conjunto completo).
- `significance_test.py` evalúa si esa mejora es estadísticamente significativa dado el tamaño del conjunto de prueba.

## Datos

- Fuente: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
  (Hou et al., 2024), categoría *Beauty and Personal Care*.
- Archivos: `raw/review_categories/Beauty_and_Personal_Care.jsonl` (reseñas) y
  `raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl` (metadatos). Los notebooks los descargan automáticamente.
- Columnas usadas: `user_id`, `parent_asin` (renombrada `product_id`), `rating`, `timestamp`; de los metadatos,
  título, tienda, precio, categorías, características y descripción.
- Tamaño: 23.911.390 reseñas (23.591.312 tras eliminar duplicados), 11,3 M usuarios y 1,03 M productos.

## Estructura del repositorio y orden de ejecución

Los notebooks son exportaciones de Databricks (`# Databricks notebook source`). Cada uno lee las tablas Delta
que escribió el anterior en `workspace.default`.

| Orden | Archivo | Qué hace | Tablas que produce |
| --- | --- | --- | --- |
| 1 | `Preproccesing.py` | Descarga, limpieza, filtro 5-core, gráficos EDA, indexado, partición temporal 80/10/10, tablas de arranque en frío | `amazon_all_data`, `amazon_filtered_data`, `amazon_train`, `amazon_val`, `amazon_test` |
| 2 | `Preprocessing_no_filter.py` | Igual sin filtro 5-core; tamaño en memoria y estadísticas descriptivas | `amazon_train_nonfilter`, `amazon_val_nonfilter`, `amazon_test_nonfilter` |
| 3 | `Analysis_experimental.py` | Exploración inicial: ALS sobre calificaciones crudas y residuales, CF, primer blend | `als_tuning1`, `als_tuning2`, `item_neighbors`, `blend_val`, `blend_test` |
| 4 | `Analysis_experimental_tuned.py` | Ajuste del baseline, malla ALS, malla CF, análisis de solapamiento, blend final en prueba | `final_item_b`, `final_user_b`, `item_neighbors`, `blend_val`, `blend_test` |
| 5 | `Analysis_experimental_no_filter.py` | Mismos experimentos sin filtro | `user_map_nf`, `item_map_nf` |
| 6 | `Analysis_content.py` / `(Clone) Analysis_content_presentationdata2.py` | Metadatos, TF-IDF, modelo de texto, prior out-of-fold (5-core) | `item_meta`, `item_features`, `item_labels`, `item_b_text` |
| 7 | `Analysis_content_presentationdata.py` | Versión de la presentación: malla de regularización, pesos de palabras, resultado en prueba | `item_meta2`, `item_labels`, `item_b_text`, `amazon_reg_validation` |
| 8 | `Analysis_content_no_filter.py` | Prior de texto sobre el conjunto completo y resultado en prueba | `item_features_nofilter`, `item_b_text_2` |
| 9 | `significance_test.py` | Prueba pareada de significancia (baseline vs. prior de texto) en 5-core y completo | `significance_results` |

`Proyecto-ST1803 (1).zip` es una copia comprimida de los mismos notebooks.

## Requisitos

- Un workspace de **Databricks** con Unity Catalog (catálogo `workspace`, esquema `default`) y cómputo con
  Spark 3.5 o superior (clúster clásico o *serverless*).
- Acceso a internet desde el clúster para descargar los datos de Hugging Face.
- Librerías: ver `requirements.txt`. En Databricks basta con `%pip install huggingface_hub`
  (ya incluido en los notebooks que descargan datos).

## Cómo ejecutar

1. En Databricks: *Workspace → Create → Git folder* y pegar la URL de este repositorio
   (o importar los `.py` como notebooks).
2. Conectar los notebooks a un clúster o cómputo *serverless*.
3. Ejecutar los notebooks en el orden de la tabla. Los pasos 1–2 crean el volumen
   `/Volumes/workspace/default/raw` y descargan los datos.
4. Los resultados se imprimen en cada notebook y quedan guardados como tablas Delta en `workspace.default`.

Tiempo aproximado: el filtro 5-core y el cálculo de pares de ítems del filtrado colaborativo son los pasos más
costosos; el resto son agregaciones y joins distribuidos.

## Metodología en breve

- **Partición temporal** por `timestamp` (80 % entrenamiento, 10 % validación, 10 % prueba). Hiperparámetros
  ajustados solo en validación; la prueba se ejecuta una vez.
- **Métrica:** RMSE con predicciones recortadas a [1, 5]. Cuando un método no puede predecir una fila
  (arranque en frío), el baseline se evalúa sobre las mismas filas.
- **Baseline:** `r̂ = μ + b_i + b_u` con sesgos regularizados (ridge, forma cerrada).
- **ALS y CF ítem–ítem:** ajustados sobre los residuales del baseline.
- **Prior de texto:** TF-IDF (20.000 palabras) + log precio → regresión ridge que predice `b_i`; se usa como
  valor hacia el que se contrae `b_i`, con predicciones out-of-fold para evitar fuga de información.

## Referencias

- Hou, Y. et al. (2024). *Bridging Language and Items for Retrieval and Recommendation.* arXiv:2403.03952.
- Koren, Y., Bell, R. y Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer.
- Leskovec, J., Rajaraman, A. y Ullman, J. D. (2020). *Mining of Massive Datasets.* Cambridge University Press.
