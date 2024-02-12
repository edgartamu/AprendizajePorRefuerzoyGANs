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

Previamente en "Aprendizaje por refuerzo", se ha introducido una de sus técnicas más populares: el *Q-learning*. Y además, se han establecido las bases hablando de procesos de decisión de Markov, políticas y funciones de valor.

En este apartado extenderemos las técnicas de ***Q-Learning*** clásico incorporando el uso de *redes neuronales*, dando pie a la evolución a los modelos de ***Deep Q Network***.

Algunos conocimientos de Deep Learning y recursos recomendados de interes:

- https://www.coursera.org/learn/neural-networks-deep-learning
- https://www.tensorflow.org/guide/core/mlp_core
- https://www.tensorflow.org/agents/tutorials/0_intro_rl?hl=es-419



---

# Recursos didácticos

1. [Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529.](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
2. [Wang, Ziyu, et al. “Dueling network architectures for deep reinforcement learning.” arXiv preprint arXiv:1511.06581 (2015)](https://arxiv.org/pdf/1511.06581.pdf)
3. [Van Hasselt, Hado, Arthur Guez, and David Silver. “Deep reinforcement learning with double q-learning.” Thirtieth AAAI conference on artificial intelligence. 2016](https://doi.org/10.1609/aaai.v30i1.10295)
4. [Tensorflow tutoriales de agentes para aprendizaje por refuerzo](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial?hl=es-419)
5. [Deep Multi-Agent Reinforcement Learning using DNN-Weight Evolution to Optimize Supply Chain Performance](https://www.researchgate.net/publication/322677430_Deep_Multi-Agent_Reinforcement_Learning_using_DNN-Weight_Evolution_to_Optimize_Supply_Chain_Performance)



