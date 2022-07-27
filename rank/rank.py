import pandas as pd


ranks = pd.read_csv('ranks.csv')
ranks.head(20)


def get_rank(surah_number, verse_number):
    try:
        return ranks[(ranks['chapter'] == surah_number) & (ranks['verse'] == verse_number)].iloc[0]['rank']
    except:
        return float('inf')


def sort_batch(batch):
    return sorted(batch, key=lambda o: get_rank(o['surah_number'], o['verse_number']))


def sort_verses(verses, batch_size):
    index = batch_size
    sorted_verses = []
    while index <= len(verses):
        sorted_verses.extend(sort_batch(verses[index - batch_size: index]))
        index += batch_size
    sorted_verses.extend(verses[index - batch_size:])
    index -= batch_size
    return sorted_verses
