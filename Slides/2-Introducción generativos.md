---
title       : MongoDB
author      : Edgar Talavera Muñoz <e.talavera@upm.es>
description : >
  En ***2015***, *DeepMind*, siendo ya parte de Google, presentó un avance en el campo del aprendizaje por refuerzo profundo con la introducción de ***Deep Q Network (DQN)***, marcando el comienzo del campo conocido hoy como *Deep Reinforcement Learning*.
marp        : true
paginate    : true
theme       : bbdd
---
<!--
_header: ''
_footer: ![height:30](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-informational.svg)<br>Esta obra está bajo una [licencia de Creative Commons Reconocimiento-NoComercial-CompartirIgual 4.0 Internacional](http://creativecommons.org/licenses/by-nc-sa/4.0/). Icono diseñado por Flaticon
-->

# Aprendizaje por refuerzo y técnicas generativas.

Deep Q Network (DQN)

---

<!-- _class: section -->
# Introducción a los modelos Generativos


---


# Introducción

Tradicionalmente existen dos paradigmas de aprendizaje y distintos tipos de modelos, como son el aprendizaje ***Supervisado*** y ***No Supervisado***

<p align="center" width="100%">
    <img width="60%" src="images/Generativos/SupvsUns.png"> 
</p>

---

<style scoped>
li { font-size: 0.8rem; }
p { font-size: 0.8rem; }
</style>

# Introducción

A su vez, existen diferentes modelos o diferentes problemas (paradigmas) que afrontar con los modelos de IA:

- **Discriminativos**: predicen la probabilidad de pertenecer a una clase dado las carácterísticas de los datos de entrada.
- **Generativos**: buscan modelar cómo se generan los datos observados y pueden generar nuevos datos similares

<p align="center" width="100%">
    <img width="50%" src="images/Generativos/DisvsGen.png"> 
</p>

---

# Introducción - paradigmas de los modelos generativos


- No solo aprenden a diferenciar, sino que aprenden la estructura de los datos
- Son útiles en aprendizaje no supervisado
- Es más sencillo obtener una idea de qué caracteriza una clase
- Son más costosos computacionalemente


---

# Introducción - paradigmas de los modelos generativos

Ejemplos y evolución de los modelos generativos:
- Naive Bayes (1960~1970)
- Máquinas de Boltzmann (RBM) (1980)
- Modelos de Markov (HMM) (1960~1970)
- Gaussian Mixture Models (GMMs) (1977)
- Autoencoders variacionales (AEs) (2013)
- Generative Adversarial Networks (GANs) (2014)
- Transformers (2017)
- Diffusion Models (2020)

---

<!-- _class: section -->
# Autoencoders - AEs

---

# ¿Qué son los Autoencoders?

Un tipo de red neuronal que puede aprender a comprimir y luego reconstruir datos.
- Un autoencoder es un tipo de red neuronal utilizada en tareas de **aprendizaje no supervisado**.
- Su objetivo es aprender una **representación compacta** de los datos de entrada.
- Consiste en dos partes principales: el ***codificador*** y el ***decodificador***.
- El objetivo principal de un autoencoder es ***minimizar la diferencia*** entre los datos de entrada y los datos reconstruidos por el decodificador. 

---

# ¿Qué son los Autoencoders?

Son redes neuronales con una arquitectura compuesta de dos componentes que se entrenan al mismo tiempo:
- **Codificador**: Transforma los datos de entrada en una representación de menor dimensión.
- **Decodificador**: Toma esta representación y reconstruye los datos originales.

<p align="center" width="100%">
    <img width="80%" src="images/Generativos/AE.png"> 
</p>

---

# Recursos didácticos

1. [Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529.](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
2. [Wang, Ziyu, et al. “Dueling network architectures for deep reinforcement learning.” arXiv preprint arXiv:1511.06581 (2015)](https://arxiv.org/pdf/1511.06581.pdf)
3. [Van Hasselt, Hado, Arthur Guez, and David Silver. “Deep reinforcement learning with double q-learning.” Thirtieth AAAI conference on artificial intelligence. 2016](https://doi.org/10.1609/aaai.v30i1.10295)
4. [Tensorflow tutoriales de agentes para aprendizaje por refuerzo](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial?hl=es-419)
5. [Deep Multi-Agent Reinforcement Learning using DNN-Weight Evolution to Optimize Supply Chain Performance](https://www.researchgate.net/publication/322677430_Deep_Multi-Agent_Reinforcement_Learning_using_DNN-Weight_Evolution_to_Optimize_Supply_Chain_Performance)



