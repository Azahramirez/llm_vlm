# Detección de huskies en imágenes usando un modelo de lenguaje visual (VLM) y anotación de cuadros delimitadores (bounding boxes).

>**Dataset**: [Husky Dataset](https://huggingface.co/datasets/mlnomad/imnet1k_Siberian_husky)

## Descripción
En la carpeta de `src`, se encuentran los scripts principales para la detección y anotación de huskies en imágenes. Estos scripts utilizan un modelo de lenguaje visual (VLM) para generar cuadros delimitadores (bounding boxes) alrededor de los huskies en las imágenes.

1. `drawing_bbox2d.py`: Este script procesa un archivo Parquet que contiene imágenes y etiquetas, y utiliza un modelo VLM para generar cuadros delimitadores alrededor de los huskies en las imágenes. Los resultados se guardan en una carpeta especificada.

2. `correction.py`: Este script permite corregir manualmente los cuadros delimitadores generados por el modelo VLM. Se puede interactuar con la imagen para agregar, eliminar o ajustar los cuadros delimitadores.

3. `validation.py`: Este script valida los cuadros delimitadores generados y corregidos, mostrando las imágenes con los cuadros delimitadores superpuestos y guardando los resultados en una carpeta de salida.


## Carpetas de salida

- En `validation`, se guardan las imágenes anotadas con los cuadros delimitadores generados y corregidos.

- En `yolo_labels2B`, se guardan los archivos de etiquetas en formato YOLO correspondientes a las imágenes anotadas.


