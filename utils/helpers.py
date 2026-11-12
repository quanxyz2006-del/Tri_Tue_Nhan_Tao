from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def bleu_scores(references, hypotheses):
    """Corpus BLEU on a 0..100 scale with method1 smoothing."""
    return {f"BLEU-{n}": 100 * corpus_bleu(
        references, hypotheses, weights=tuple([1 / n] * n),
        smoothing_function=SmoothingFunction().method1) for n in range(1, 5)}
