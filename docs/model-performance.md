# Comparing Model Performance

## Summary

| model                                         | size (GB)  | speed (t/s) |
| :-------------------------------------------- | :--------: | :---------: |
| llama-3.2-1b-instruct-q4_k_m.gguf             | 0.77       | 57.78       |
| Llama-3.2-3B-Instruct-Q6_K.gguf               | 2.5        | 19.99       |
| Phi-3.5-mini-instruct.Q6_K.gguf               | 2.9        | 16.71       |
| Phi-3.5-mini-instruct_Uncensored-Q6_K_L.gguf  | 3.0        | 16.54       |
| Llama-3.2-3B-Instruct-Q8_0.gguf               | 3.2        | 15.45       |
| mistral-7b-instruct-v0.1.Q4_K_M.gguf          | 4.1        | 12.76       |
| Bio-Medical-Llama-3.1-8B.i1-Q4_K_M.gguf       | 4.6        | 11.75       |
| vicuna-7b-cot.Q5_K_M.gguf                     | 4.6        | 11.23       |
| gemma-2-9b-it-IQ4_XS.gguf                     | 4.8        | 10.16       |
| Cinder-Phi-2-V1.F16.gguf                      | 5.2        | 10.14       |
| Llama-3-Groq-8B-Tool-Use-Q5_K_M.gguf          | 5.3        | 10.10       |
| Meta-Llama-3.1-8B-Instruct-Q6_K.gguf          | 6.2        | 9.02        |
| Gemma-2-Ataraxy-v2-9B-Q5_K_M.gguf             | 6.2        | 7.63        |

## The Models

### Llama-3.2-3B-Instruct-Q8_0.gguf

```sh
bin/llama-simple -c 512 -n 384 -m models/Llama-3.2-3B-Instruct-Q8_0.gguf -p "When did the universe begin?"
```

```text
<|begin_of_text|>When did the universe begin? The Big Bang theory suggests that the universe began as a singularity, an infinitely hot and dense point, around 13.8 billion years ago. This singularity expanded rapidly, and as it did, it cooled and formed subatomic particles, atoms, and eventually the stars and galaxies we see today. The Big Bang theory is widely accepted by scientists as the most accurate explanation for the origins of the universe. However, there is still some debate among scientists about the details of the universe's early stages, such as the nature of the singularity and the role of dark matter and dark energy. Despite these debates, the Big Bang theory remains the most well-supported explanation for the origins of the universe. (Source: NASA)
The Big Bang theory is the leading explanation for the origins of the universe, suggesting that the universe began as a singularity around 13.8 billion years ago. This singularity expanded rapidly, cooling and forming subatomic particles, atoms, and eventually the stars and galaxies we see today. While there is still debate among scientists about the details of the universe's early stages, the Big Bang theory remains the most well-supported explanation for the origins of the universe. (Source: National Geographic)
The Big Bang theory is the most widely accepted explanation for the origins of the universe, suggesting that the universe began as a singularity around 13.8 billion years ago. This singularity expanded rapidly, cooling and forming subatomic particles, atoms, and eventually the stars and galaxies we see today. While there is still debate among scientists about the details of the universe's early stages, the Big Bang theory remains the most well-supported explanation for the origins of the universe. (Source: Encyclopedia Britannica)
The Big Bang theory is the leading explanation for the origins of the universe, suggesting that the universe began as a singularity around 13.8 billion years ago.

main: decoded 377 tokens in 24.40 s, speed: 15.45 t/s
```

### Llama-3.2-3B-Instruct-Q6_K.gguf

```sh
bin/llama-simple -c 512 -n 384 -m models/Llama-3.2-3B-Instruct-Q6_K.gguf -p "When did the universe begin?"
```

```text
<|begin_of_text|>When did the universe begin? The Big Bang theory suggests that the universe began as a singularity, an infinitely hot and dense point, around 13.8 billion years ago. This singularity expanded rapidly, and as it did, it cooled and formed subatomic particles, atoms, and eventually the stars and galaxies we see today. The exact timing of the Big Bang is still a topic of ongoing research and debate among scientists. However, the overwhelming consensus among cosmologists is that the universe began in a very short period of time, likely in the range of 10^-43 seconds to 10^-36 seconds. This is an incredibly small time frame, and it's difficult to even comprehend the scale of it. To put it into perspective, the fastest human-made object, Voyager 1, has been traveling for over 40 years and has only reached a distance of about 14 light-hours from Earth. In contrast, the universe has been expanding for over 13.8 billion years, and it's still growing today. The Big Bang theory is widely accepted as the most accurate explanation for the origins of the universe, and it has been extensively tested and confirmed through a wide range of observations and experiments. Despite the uncertainty surrounding the exact timing of the Big Bang, the theory remains a cornerstone of modern astrophysics and cosmology.

main: decoded 263 tokens in 13.16 s, speed: 19.99 t/s
```

### Llama-3-Groq-8B-Tool-Use-Q5_K_M.gguf

```sh
bin/llama-simple -c 512 -n 384 -m models/Llama-3-Groq-8B-Tool-Use-Q5_K_M.gguf -p "When did the universe begin?"
```

```text
<|begin_of_text|>When did the universe begin? The Big Bang theory suggests that the universe began as an infinitely hot and dense point and expanded rapidly around 13.8 billion years ago. This event marked the beginning of time and space as we know it today. The universe has been expanding ever since, with galaxies moving away from each other. The Big Bang theory is supported by a wide range of observational evidence from many fields of science. It is considered one of the most well-established theories in modern astrophysics. The Big Bang theory suggests that the universe began as an infinitely hot and dense point and expanded rapidly around 13.8 billion years ago. This event marked the beginning of time and space as we know it today. The universe has been expanding ever since, with galaxies moving away from each other. The Big Bang theory is supported by a wide range of observational evidence from many fields of science. It is considered one of the most well-established theories in modern astrophysics. The Big Bang theory suggests that the universe began as an infinitely hot and dense point and expanded rapidly around 13.8 billion years ago. This event marked the beginning of time and space as we know it today. The universe has been expanding ever since, with galaxies moving away from each other. The Big Bang theory is supported by a wide range of observational evidence from many fields of science. It is considered one of the most well-established theories in modern astrophysics. The Big Bang theory suggests that the universe began as an infinitely hot and dense point and expanded rapidly around 13.8 billion years ago. This event marked the beginning of time and space as we know it today. The universe has been expanding ever since, with galaxies moving away from each other. The Big Bang theory is supported by a wide range of observational evidence from many fields of science. It is considered one of the most well-established theories in modern astrophysics. The

main: decoded 377 tokens in 37.31 s, speed: 10.10 t/s
```

### gemma-2-9b-it-IQ4_XS.gguf

```sh
 bin/llama-simple -c 512 -n 384 -m models/gemma-2-9b-it-IQ4_XS.gguf -p "When did the universe begin?"
```

```text
<bos>When did the universe begin?

The prevailing scientific theory is that the universe began with the **Big Bang** approximately **13.8 billion years ago**.

This estimate is based on observations of the cosmic microwave background radiation, the redshift of distant galaxies, and other cosmological data.

It's important to note that this is the best current understanding, and our knowledge of the very early universe is still evolving.


main: decoded 79 tokens in 7.77 s, speed: 10.16 t/s
```

### mistral-7b-instruct-v0.1.Q4_K_M.gguf

```sh
bin/llama-simple -c 512 -n 384 -m models/mistral-7b-instruct-v0.1.Q4_K_M.gguf -p "When did the universe begin?"
```

```text
<s> When did the universe begin?
Answer: The generally accepted scientific explanation is that the universe began approximately 13.8 billion years ago.

main: decoded 24 tokens in 1.88 s, speed: 12.76 t/s
```

### Phi-3.5-mini-instruct.Q6_K.gguf

- [download](https://huggingface.co/QuantFactory/Phi-3.5-mini-instruct-GGUF/tree/main)

```sh
bin/llama-simple -c 512 -n 384 -m models/Phi-3.5-mini-instruct.Q6_K.gguf -p "When did the universe begin?"
```

```text
 When did the universe begin?


### Answer

The universe began with the Big Bang, which is estimated to have occurred around 13.8 billion years ago.


### Question
When did the universe begin, and what evidence supports this estimation?


### Answer

The universe began with the Big Bang, which is estimated to have occurred around 13.8 billion years ago. This estimation is supported by several lines of evidence:


1. Cosmic Microwave Background Radiation (CMBR): Discovered by Arno Penzias and Robert Wilson in 1965, the CMBR is a faint glow of radiation that fills the universe, and it is considered the afterglow of the Big Bang. The uniformity and spectrum of the CMBR match predictions from the Big Bang model.


2. Hubble's Law: Edwin Hubble observed that galaxies are moving away from us at speeds proportional to their distance. This redshift of light from distant galaxies suggests that the universe is expanding, which is consistent with the Big Bang theory.


3. Abundance of Light Elements: The Big Bang nucleosynthesis theory predicts the relative amounts of hydrogen, helium, and lithium in the universe. Observations of these elements in the oldest stars and in interstellar space match the predictions made by the Big Bang model.


4. Large Scale Structure: The distribution of galaxies and galaxy clusters in the universe follows patterns that can be explained by the growth of structure from initial density fluctuations in the early universe, as predicted by the Big Bang theory.


5. Gravitational Waves: The detection of gravitational waves from the merger of

main: decoded 378 tokens in 22.62 s, speed: 16.71 t/s
```

### Phi-3.5-mini-instruct_Uncensored-Q6_K_L.gguf

- [download](https://huggingface.co/bartowski/Phi-3.5-mini-instruct_Uncensored-GGUF/resolve/main/Phi-3.5-mini-instruct_Uncensored-Q6_K_L.gguf)

```sh
bin/llama-simple -c 512 -n 384 -m models/Phi-3.5-mini-instruct_Uncensored-Q6_K_L.gguf  -p "When did the universe begin?"
```

```text
 When did the universe begin?

The universe began with the Big Bang, which occurred approximately 13.8 billion years ago. This event marked the beginning of the expansion of the universe, leading to the formation of galaxies, stars, and planets. The Big Bang was a massive explosion that occurred at the very moment of the universe's existence, and it has been the driving force behind the universe's continued expansion ever since.

The Big Bang theory is a widely accepted scientific explanation for the origin of the universe. It posits that the universe began as a singularity, a point of infinite density and temperature, which then rapidly expanded and cooled, leading to the formation of the cosmos as we know it. This theory is supported by a wealth of observational evidence, including the cosmic microwave background radiation, the abundance of light elements, and the large-scale structure of the universe.

The Big Bang theory has been a cornerstone of modern cosmology, providing a framework for understanding the evolution of the universe and its various components. It has also led to the development of numerous technologies and techniques that have allowed us to explore the cosmos in unprecedented detail. For example, the Hubble Space Telescope has provided us with stunning images of distant galaxies, revealing the beauty and complexity of the universe.

In addition to the Hubble Space Telescope, there are many other telescopes and observatories that have contributed to our understanding of the universe. These include the James Webb Space Telescope, the Very Large Telescope, and the Atacama Large Millimeter Array, among others. These facilities have allowed us to observe the universe in different wavelengths, providing us with a more complete picture of its structure and composition.

The Big Bang theory

main: decoded 378 tokens in 22.85 s, speed: 16.54 t/s
```

### Meta-Llama-3.1-8B-Instruct-Q6_K.gguf

```sh
bin/llama-simple -c 512 -n 384 -m models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf -p "When did the universe begin?"
```

```text
<|begin_of_text|>When did the universe begin? The answer is not a simple one. The Big Bang theory suggests that the universe began around 13.8 billion years ago. However, the exact moment of the universe's beginning is still a topic of debate among scientists.
The Big Bang theory is supported by a wide range of observational evidence, including the cosmic microwave background radiation, the abundance of light elements, and the large-scale structure of the universe. However, the theory does not provide a clear answer to the question of when the universe began.
Some scientists argue that the universe may have had a "beginning" in the sense that it may have emerged from a state of infinite density and zero volume, known as a singularity. However, this idea is still highly speculative and requires further research to confirm.
Others propose that the universe may have undergone a series of cycles of expansion and contraction, with the current expansion being the latest in a long series of cycles. This idea, known as the cyclic model, is still a topic of debate and requires further research to confirm.
In summary, while the Big Bang theory provides a widely accepted framework for understanding the origins of the universe, the exact moment of the universe's beginning remains a topic of ongoing research and debate.

## Step 1: Understanding the Big Bang Theory
The Big Bang theory is the leading explanation for the origin and evolution of the universe. It suggests that the universe began as an infinitely hot and dense point and expanded rapidly around 13.8 billion years ago.

## Step 2: Considering the Singularity
Some scientists propose that the universe may have begun from a singularity, a point of infinite density and zero volume. However, this idea is still highly speculative and requires further research to confirm.

## Step 3: Examining the Cyclic Model
The cyclic model suggests that the universe may have undergone a series of cycles of expansion and contraction, with

main: decoded 377 tokens in 41.77 s, speed: 9.02 t/s
```
