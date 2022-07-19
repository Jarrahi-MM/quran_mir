from preprocess_quran_text import quran_series, quran_normalizer
from tfidf_vectorizer import get_most_similars


class TestTfIdfRetrieval:
    def q1(self):
        query = 'الحمد لله'
        true_responses = [
            'الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
            'وَ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
            'الْحَمْدُ لِلَّهِ الَّذِي لَهُ مَا فِي السَّمَاوَاتِ وَ مَا فِي الْأَرْضِ وَ لَهُ الْحَمْدُ فِي الْآخِرَةِ وَ هُوَ الْحَكِيمُ الْخَبِيرُ',
            'فَقُطِعَ دَابِرُ الْقَوْمِ الَّذِينَ ظَلَمُوا وَ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
        ]
        responses = list(get_most_similars(quran_series, quran_normalizer(query), 10)['آیه'])
        for r in true_responses:
            assert (r in responses)
