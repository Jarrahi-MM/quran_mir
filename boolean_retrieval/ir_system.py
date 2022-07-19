import collections

from boolean_retrieval.boolean import BooleanModel
from preprocess_quran_text import quran_normalizer
from nltk.stem.isri import ISRIStemmer

st = ISRIStemmer()


class IRSystem:

    def __init__(self, docs=None):
        self._docs = docs
        self._inverted_index = self._preprocess_corpus()

    def _preprocess_corpus(self):
        index = {}
        for i, doc in enumerate(self._docs):
            for token in doc.split():
                if token not in index.keys():
                    index[token] = [i]
                else:
                    index[token].append(i)
        return index

    def _get_posting_list(self, word):
        return [doc_id for doc_id in self._inverted_index[word] if doc_id is not None]

    @staticmethod
    def _parse_query(infix_tokens, query_type):
        """ Parse Query 
        Parsing done using Shunting Yard Algorithm 
        """
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        output = []
        operator_stack = []

        for token in infix_tokens:
            # if operator, pop operators from operator stack to queue if they are of higher precedence
            if token in precedence:
                # if operator stack is not empty
                if operator_stack:
                    current_operator = operator_stack[-1]
                    while operator_stack and precedence[current_operator] > precedence[token]:
                        output.append(operator_stack.pop())
                        if operator_stack:
                            current_operator = operator_stack[-1]
                operator_stack.append(token)  # add token to stack
            else:
                if query_type == "complete":
                    output.append(quran_normalizer(token))
                elif query_type == "lemma":
                    output.append(quran_normalizer(token))  # Todo find arabic lemmarizer
                elif query_type == "root":
                    output.append(st.stem(quran_normalizer(token)))

        # while there are still operators on the stack, pop them into the queue
        while operator_stack:
            output.append(operator_stack.pop())

        return output

    def process_query(self, query, query_type):
        # prepare query list
        # query = query.split(' ')

        indexed_doc_ids = list(range(1, len(self._docs) + 1))

        results_stack = []
        postfix_queue = collections.deque(
            self._parse_query(query, query_type))  # get query in postfix notation as a queue

        while postfix_queue:
            token = postfix_queue.popleft()
            result = []  # the evaluated result at each stage
            # if operand, add postings list for term to results stack
            if token != 'AND' and token != 'OR' and token != 'NOT':
                # default empty list if not in dictionary
                if token in self._inverted_index:
                    result = self._get_posting_list(token)

            elif token == 'AND':
                right_operand = results_stack.pop()
                left_operand = results_stack.pop()
                result = BooleanModel.and_operation(left_operand, right_operand)  # evaluate AND

            elif token == 'OR':
                right_operand = results_stack.pop()
                left_operand = results_stack.pop()
                result = BooleanModel.or_operation(left_operand, right_operand)  # evaluate OR

            elif token == 'NOT':
                right_operand = results_stack.pop()
                result = BooleanModel.not_operation(right_operand, indexed_doc_ids)  # evaluate NOT

            results_stack.append(result)

            # NOTE: at this point results_stack should only have one item and it is the final result
        if len(results_stack) != 1:
            print("ERROR: Invalid Query. Please check query syntax.")  # check for errors
            return None

        return results_stack.pop()
