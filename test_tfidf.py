from preprocess_quran_text import quran_series, quran_normalizer
from quran_ir import TfIdfQuranIR


class TestTfIdfRetrieval:
    tfidf_quran_ir = TfIdfQuranIR()

    def test_q1(self):
        query = 'الحمد لله'
        true_responses = [
            'الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
            'وَ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
            'الْحَمْدُ لِلَّهِ الَّذِي لَهُ مَا فِي السَّمَاوَاتِ وَ مَا فِي الْأَرْضِ وَ لَهُ الْحَمْدُ فِي الْآخِرَةِ وَ هُوَ الْحَكِيمُ الْخَبِيرُ',
            'فَقُطِعَ دَابِرُ الْقَوْمِ الَّذِينَ ظَلَمُوا وَ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ',
        ]
        responses = list(
            TestTfIdfRetrieval.tfidf_quran_ir.get_most_similars(quran_series, quran_normalizer(query), 10)['آیه'])
        for r in true_responses:
            assert (r in responses)

    def test_q2(self):
        query = 'فرعون موسی'
        true_responses = [
            'وَ قَالَ مُوسَىٰ يَا فِرْعَوْنُ إِنِّي رَسُولٌ مِنْ رَبِّ الْعَالَمِينَ',
            'إِلَىٰ فِرْعَوْنَ وَ مَلَئِهِ فَاتَّبَعُوا أَمْرَ فِرْعَوْنَ وَ مَا أَمْرُ فِرْعَوْنَ بِرَشِيدٍ',
            'نَتْلُو عَلَيْكَ مِنْ نَبَإِ مُوسَىٰ وَ فِرْعَوْنَ بِالْحَقِّ لِقَوْمٍ يُؤْمِنُونَ',
            'وَ فِي مُوسَىٰ إِذْ أَرْسَلْنَاهُ إِلَىٰ فِرْعَوْنَ بِسُلْطَانٍ مُبِينٍ',
        ]
        responses = list(
            TestTfIdfRetrieval.tfidf_quran_ir.get_most_similars(quran_series, quran_normalizer(query), 10)['آیه'])
        for r in true_responses:
            assert (r in responses)

    def test_q3(self):
        query = 'یوم القیامه'
        true_responses = [
            'يَسْأَلُ أَيَّانَ يَوْمُ الْقِيَامَةِ',
            'ثُمَّ إِنَّكُمْ يَوْمَ الْقِيَامَةِ تُبْعَثُونَ',
            'ثُمَّ إِنَّكُمْ يَوْمَ الْقِيَامَةِ عِنْدَ رَبِّكُمْ تَخْتَصِمُونَ',
            'خَالِدِينَ فِيهِ وَ سَاءَ لَهُمْ يَوْمَ الْقِيَامَةِ حِمْلًا',
        ]
        responses = list(
            TestTfIdfRetrieval.tfidf_quran_ir.get_most_similars(quran_series, quran_normalizer(query), 10)['آیه'])
        for r in true_responses:
            assert (r in responses)
