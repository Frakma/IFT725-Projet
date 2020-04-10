class DataCreator:
    """Class with create training/test data with labels
    """
    def __init__(self, sentences, length_train):
        self.sentences = sentences
        self.length_train = length_train

    def get_data_from_sentence(self, sentence):
        """Function which create data tokens and labels

            Parameters:
            sentence (list): list of words

            Returns:
            tuple:data and labels
        """
        if len(sentence) < self.length_train + 1:
            return None, None

        data = []
        target = []

        index = 0
        while self.length_train + index < len(sentence):
            data_tmp = sentence[index:index + self.length_train]
            target_tmp = sentence[index + self.length_train]

            data.append(data_tmp)
            target.append(target_tmp)

            index += 1

        return data, target

    def tokenize_sentences(self):
        """Function which create data tokens and labels

            Parameters:
            sentences (list): list of sentences of words

            Returns:
            tuple:data and labels
        """
        data = []
        labels = []

        for sentence in self.sentences:
            data_tmp, label_tmp = self.get_data_from_sentence(sentence)

            if data_tmp != None and label_tmp != None:
                data.extend(data_tmp)
                labels.extend(label_tmp)

        return data, labels