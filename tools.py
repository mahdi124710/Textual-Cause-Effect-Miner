import torch


def run_model(model, tokenizer, sentence, question):
    """

    :param model: The pre-trained model
    :param tokenizer: The tokenizer loaded from pre-trained model
    :param sentence: The raw sentence you want to find marker, cause, and effect in
    :param question: If you want to find cause or effect this should be the found marker. if you want to find marker
    set this the same as the question in training data.
    :return: This would return a substring of sentence showing marker, cause, or effect
    """

    inputs = tokenizer.encode_plus(question, sentence, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs)[0], model(**inputs)[1]

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    return answer