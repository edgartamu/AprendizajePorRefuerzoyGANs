---
marp: true
size: 4:3
auto-scaling: 
    - true
    - fittingHeader
    - math
    - code
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.jpg')
paginate: true
header: Aprendizaje por refuerzo y técnicas generativas. 
---
<!--
_header: ''
_footer: ![height:30](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-informational.svg)<br>Esta obra está bajo una [licencia de Creative Commons Reconocimiento-NoComercial-CompartirIgual 4.0 Internacional](http://creativecommons.org/licenses/by-nc-sa/4.0/). Icono diseñado por Flaticon
-->

# Aprendizaje por refuerzo y técnicas generativas.

Deep Q Network (DQN)

---

<style scoped>
li { font-size: 0.7rem; }
p { font-size: 0.75rem; }
</style>

## Introducción

Visto anteriormente el Aprendizaje por refuerzo, e introducido a este campo a través de una de sus técnicas más populares: el *Q-learning*. Y además, se han establecido las bases hablando de procesos de decisión de Markov, políticas y funciones de valor.

En este apartado extenderemos las técnicas de ***Q-Learning*** clásico incorporando el uso de *redes neuronales*, dando pie a la evolución a los modelos de ***Deep Q Network***.

Si quieres ahondar en conceptos de Deep Learning, algunos recursos recomendados:

- https://www.coursera.org/learn/neural-networks-deep-learning
- https://www.tensorflow.org/guide/core/mlp_core
- https://www.tensorflow.org/agents/tutorials/0_intro_rl?hl=es-419



---

<style scoped>
li { font-size: 0.6rem; }
p { font-size: 0.7rem; }
</style>

## Un poco de historia

En ***2015***, *DeepMind*, siendo ya parte de Google, presentó un avance en el campo del aprendizaje por refuerzo profundo con la introducción de ***Deep Q Network (DQN)***, marcando el comienzo del campo conocido hoy como *Deep Reinforcement Learning*.

***DQN*** cambia la forma en que se aborda el aprendizaje automático en entornos de toma de decisiones secuenciales. 
   - En lugar de procesar datos estáticos (imágenes o texto), el aprendizaje por refuerzo implica aprender a tomar decisiones secuenciales mientras se maximiza una recompensa acumulativa.
   - ***DQN*** se lanzó en un videojuego de ***Atari***, utilizando solo píxeles de la pantalla y retroalimentación de recompensa sencilla, logrando superar el rendimiento de los "expertos/frikies" en varios juegos, incluidos títulos icónicos como Breakout, Space Invaders y Pong. 
   - La capacidad de ***DQN*** radica aprender directamente de la experiencia, o más bien del escenario que se crea en cada instantem, y tomar decisiones óptimas en entornos complejos.
   - Todo esto allanó el camino para aplicaciones en campos como la robótica, los sistemas de control, la gestión de inventario, los vehículos autónomos y más. 
---

<style scoped>
li { font-size: 0.7rem; }
p { font-size: 0.7rem; }
</style>

## Q-Learning vs. Deep Q-Learning


<p align="center" width="100%">
    <img width="60%" src="Q-learn_DQlearn.png"> 
</p>

- En ***Q-Learning*** usamos la tablas de estados y acciones, o Q-valores.
- En ***Deep Q-Learning*** utilizamos una red neuronal que aproxima o minimiza la función que aproxima esos Q-valores.

---


<style scoped>
li { font-size: 0.3rem; }
p { font-size: 0.65rem; }
</style>

## Q-Learning vs. Deep Q-Learning

El uso de ***redes neuronales*** tiene varios propósitos importantes en comparación con el enfoque tradicional de Q-Learning:

 - ***Aproximación de la función Q***: En Q-Learning la tabla que almacena los Q-valores de cada par estado-acción posible, aciendo dependiente de la complejidad del problema. Una ***red neuronal*** permite aproximar la función Q, lo que significa que ***cedemos*** a la red el aprender una representación de la función Q, como si generase su propia Q-tabla.

 - ***Generalización y abstracción***: Las redes neuronales por definición buscan generalizar a partir de ejemplos y extraer características relevantes de los datos de entrada.

 - ***Escalabilidad***: en Q-Learning vemos limitadas nuestras acciones por como de compleja podamos generar nuestra Q-tabla, las redes neuronales permiten manejar datos de alta dimensionalidad y extraer características relevantes para la toma de decisiones.

 - ***Aprendizaje de representaciones jerárquicas***: aprenden representaciones jerárquicas de los datos, lo que les permite capturar características tanto a nivel bajo (como bordes y texturas en una imagen) como a nivel alto (como objetos y patrones).



---

### Otros conocimientos recomendados

- Bases de datos relacionales
- Programación

---

<style scoped>
li { font-size: 0.8rem; }
p { font-size: 0.8rem; }
</style>

## Competencias

**CB03** |  Que los estudiantes tengan la capacidad de reunir e interpretar datos relevantes (normalmente dentro de su área de estudio) para emitir juicios que incluyan una reflexión sobre temas relevantes de índole social, científica o ética.

**CB04** | Que los estudiantes puedan transmitir información, ideas, problemas y soluciones a un público tanto especializado como no especializado.

**CE05** | Capacidad de diseñar e implementar los procesos de selección, limpieza, transformación, integración y verificación de la calidad de los datos de cara a su posterior tratamiento.

**CE07** | Capacidad de diseñar e implementar sistemas de información (incluyendo modelos de datos y estrategias de gestión de datos) dimensionados para gestionar el volumen, velocidad y variedad de los datos, de forma adecuada para su almacenamiento, procesamiento y acceso para tratamientos posteriores.


---

<style scoped>
li { font-size: 0.8rem; }
p { font-size: 0.8rem; }
</style>

## Resultados de aprendizaje

**RA129** - Ser capaz de implementar y gestionar una base de datos en un gestor no relacional.

**RA89** - Usar lenguajes de programación y de descripción de datos, comunes en Ciencia de Datos.

**RA76 - RA-APID-5** Configuración, administración, uso y optimización de sistemas gestores de bases de datos relacionales.

**RA77 - RA-APID-6** Diseño, creación, consulta y manipulación de repositorios de datos, e integración con aplicaciones del sistema.

**RA134 - RA114 - RA-APID-18** Ser capaz de utilizar las tecnologías de información para preparar los conjuntos de datos.

---

<style scoped>
li { font-size: 0.5rem; }
p { font-size: 0.5rem; }
</style>

## Temario guía docente

1. Bloque I: Bases de datos avanzadas
   1. Bases de datos basadas en ficheros
   2. Bases de datos federadas
2. Bloque II: Gobernanza de datos
   1. Gobernanza de datos
   2. Data provenance
   3. Curación y limpieza de datos
   4. Transformación y serialización de datos
3. Bloque III: Bases de datos NoSQL
   1. Fundamentos de bases de datos NoSQL
   2. Comparativa de bases de datos relacionales vs NoSQL
   3. Tipos y características de bases de datos NoSQL
   4. Bases de datos documentales
   5. Bases de datos basadas en grafos
   6. Bases de datos basadas en clave-valor
   7. Otras bases de datos (dispersas, columnares, en memoria, ...)
4. Bloque IV: Data streaming
   1. Procesamiento de datos continuos en memoria
   2. Motores de data streaming


---


<style scoped>
li { font-size: 0.6rem;}
p { font-size: 0.6rem;}
</style>

## Temario

1. Bloque I: Bases de datos avanzadas
   1. Introducción a las bases de datos NoSQL (tipos de bases de datos No-SQL)
   2. Comparativa entre bases de datos relacionales vs bases de datos NoSQL
   3. Tipos de características de bases de datos NoSQL (Bases de datos basadas en ficheros)
2. Bloque II: Gobernanza de datos
   1. Gobernanza de datos y data provenance
   2. Curación y limpieza de datos
   3. Transformación y serialización de datos
3. Bloque III: Bases de datos NoSQL
   1. Mongodb - bases de datos basadas en documentos
   2. Cassandra - bases de datos basadas en columnas (key-value)
   3. Neo4j - bases de datos basadas en grafos
4. Bloque IV: Data streaming
   1. Programación contra bases de datos
   2. Motores de data streaming 


---


## Horario

![w:500](images/Cronograma.png)

---

<style scoped>
li { font-size: 0.8rem; }
p { font-size: 0.8rem; }
</style>

## Evaluación progresiva

Prácticas (40%)
- En grupos
- Evaluación en el aula

Examen final (60%)
- Nota mínima de 4
- Todo el temario
- Viernes 9 de junio de 2023 a las 10:00 en el aula 1302

Será necesario alcanzar una nota total >= 5 para aprobar la asignatura.

---

## Evaluación prueba global

Prueba escrita el **viernes 9 de junio de 2023 a las 10:00** en el aula **1302** incluyendo preguntas teórico-prácticas de todo el temario de la asignatura.

Es necesaro obtener una nota mínima de 4 sobre 10.

Para aprobar sin haber seguido la evaluación progresiva será necesario obtener una nota total >= 5 (la prueba global cuenta un 60% del total de la nota, por lo que sacar un 5 en esta prueba no equivale a aprobar la asignatura).

---

## Convocatoria extraordinaria

Prueba escrita el **jueves 6 de julio de 2023 a las 10:00** incluyendo preguntas teórico-prácticas de todo el temario de la asignatura.

Es obligatorio alcanzar una nota mínima de 5 puntos sobre 10.

---
<style scoped>
li { font-size: 0.7rem; }
p { font-size: 0.8rem; }
</style> 

## Recursos didácticos

1. **Moodle de la asignatura.**
2. Data Governance: How to Design, Deploy, and Sustain an Effective Data Governance Program (English Edition) 2o Edición. John Ladley.
3. Data Provenance A Complete Guide - 2020 Edition (English Edition). Gerardus Blokdyk
4. Pandas 1.x Cookbook: Practical recipes for scientific computing, time series analysis, and  exploratory data analysis using Python, 2nd Edition (English Edition). Matt Harrison; Theodore Petrou
5. SQL & NoSQL Databases: A Comprehensive Introduction To Relational (SQL) And Non-relational (NoSQL) Databases (English Edition). Danniella Hardin
6. SQL & NoSQL Databases: Models, Languages, Consistency Options and Architectures for Big Data Management (English Edition). Andreas Meier; Michael Kaufmann
7. Stream Processing with Apache Flink: Fundamentals, Implementation, and Operation of Streaming Applications. Fabian Hueske, Vasiliki Kalavri



