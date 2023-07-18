# red-team-LM-for-uncertainty

Following Perez et. al [1] and using the new Semantic Entropy metric [2], we attempt to train a red-team a language model to be able to prompt another language model to produce uncertain outputs. To quantify our results, we use a question-generation language model [3] and generate questions based on (context, answer) pairs from SQuAD 1.0 [4].

Citations:
[1] Perez, Ethan, et al. "Red teaming language models with language models." arXiv preprint arXiv:2202.03286 (2022).
[2] Kuhn, Lorenz, Yarin Gal, and Sebastian Farquhar. "Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation." arXiv preprint arXiv:2302.09664 (2023).
[3] https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap
[4] Rajpurkar, Pranav, et al. "Squad: 100,000+ questions for machine comprehension of text." arXiv preprint arXiv:1606.05250 (2016).
