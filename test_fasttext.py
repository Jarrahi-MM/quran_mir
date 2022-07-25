from preprocess_quran_text import quran_normalizer, quran_series
from tools import get_most_similars
from quran_ir import FasttextQuranIR


class TestFasttextRetrieval:
    fasttext_quran_ir = FasttextQuranIR()

    def test_q1(self):
        query = 'الحمد لله'
        true_responses = [
            'الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
            'وَ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
            'الْحَمْدُ لِلَّهِ الَّذِي لَهُ مَا فِي السَّمَاوَاتِ وَ مَا فِي الْأَرْضِ وَ لَهُ الْحَمْدُ فِي الْآخِرَةِ وَ هُوَ الْحَكِيمُ الْخَبِيرُ',
            'فَلِلَّهِ الْحَمْدُ رَبِّ السَّمَاوَاتِ وَ رَبِّ الْأَرْضِ رَبِّ الْعَالَمِينَ',
        ]
        query_vec = TestFasttextRetrieval.fasttext_quran_ir.sent_to_vec(quran_normalizer(query))
        responses = list(
            get_most_similars(quran_series, TestFasttextRetrieval.fasttext_quran_ir.merged_corpus_embeddings, query_vec,
                              10)['آیه'])

        for r in true_responses:
            assert (r in responses)

    def test_q2(self):
        query = 'فرعون موسی'
        true_responses = [
            'وَ قَالَ مُوسَىٰ يَا فِرْعَوْنُ إِنِّي رَسُولٌ مِنْ رَبِّ الْعَالَمِينَ',
            'وَ فِي مُوسَىٰ إِذْ أَرْسَلْنَاهُ إِلَىٰ فِرْعَوْنَ بِسُلْطَانٍ مُبِينٍ',
            'وَ لَقَدْ أَرْسَلْنَا مُوسَىٰ بِآيَاتِنَا إِلَىٰ فِرْعَوْنَ وَ مَلَئِهِ فَقَالَ إِنِّي رَسُولُ رَبِّ الْعَالَمِينَ',
        ]
        query_vec = TestFasttextRetrieval.fasttext_quran_ir.sent_to_vec(quran_normalizer(query))
        responses = list(
            get_most_similars(quran_series, TestFasttextRetrieval.fasttext_quran_ir.merged_corpus_embeddings, query_vec,
                              10)['آیه'])
        for r in true_responses:
            assert (r in responses)

    def test_q3(self):
        query = 'یوم القیامه'
        true_responses = [
            'ثُمَّ إِنَّكُمْ يَوْمَ الْقِيَامَةِ تُبْعَثُونَ',
            'خَالِدِينَ فِيهِ وَ سَاءَ لَهُمْ يَوْمَ الْقِيَامَةِ حِمْلًا',
            'إِنَّ رَبَّكَ هُوَ يَفْصِلُ بَيْنَهُمْ يَوْمَ الْقِيَامَةِ فِيمَا كَانُوا فِيهِ يَخْتَلِفُونَ',
        ]
        query_vec = TestFasttextRetrieval.fasttext_quran_ir.sent_to_vec(quran_normalizer(query))
        responses = list(
            get_most_similars(quran_series, TestFasttextRetrieval.fasttext_quran_ir.merged_corpus_embeddings, query_vec,
                              10)['آیه'])
        for r in true_responses:
            assert (r in responses)
