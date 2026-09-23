def get_question_text(row: dict) -> str:
    """
    Given a row of the dataset, return the question text.

    Args:
        row (dict): A dictionary representing a row of the dataset.

    Returns:
        str: The question text.
    """
    a = row['choice_a']
    b = row['choice_b']
    c = row['choice_c']
    d = row['choice_d']
    q = row['question']

    question = f"Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nChoose the most suitable answer from options A, B, C, and D. You must respond with a single letter: A, B, C, or D.\n"
 
    """ 
    Original:
        Choose the most suitable answer from options A, B, C, and D. You must respond with only A, B, C, or D.
        Question: {question}
        A. {choice_a}
        B. {choice_b}
        C. {choice_c}
        D. {choice_d}
    """
    return question

def get_prediction(response: str) -> str:
    """
    Given a response string, return the predicted answer letter. MMSU answer parsing style is used.

    Args:
        response (str): The response string.

    Returns:
        str: The predicted answer letter.
    """
    if not response: # checks both None or ""
        return None
    predict = response.strip().replace('\n', '')
    if not predict:  # response was non-empty but pure whitespace/newlines
        return None
    predict = response.strip().replace('\n', '') #remove newlines and leading/trailing whitespace
    if predict[0] in 'ABCD':
        model_predict = predict[0]
    elif len(predict) > 1 and predict[-2] in 'ABCD':      # catches "The answer is A." (period is last char)
        model_predict = predict[-2]
    else:
        model_predict = None                   # wrong-format response
    return model_predict

def extract_answer_obqa(response):
    """Ported verbatim from VoiceBench's own MCQEvaluator.extract_answer
    (src/evaluator/mcq.py) — this is the exact parser their reported
    OBQA/MMSU-via-VoiceBench numbers are computed with, so using it keeps
    your results comparable to published baselines."""
    response = response.lower()
    if response.startswith('<1>') or response.startswith('<2>') or response.startswith('<3>'):
        response = response[3:].strip()
    for template in [
        "答案是[CHOICE]", "答案是 [CHOICE]", "答案是选项[CHOICE]", "答案应该是[CHOICE]",
        "答案应该是 [CHOICE]", "答案就是选项[CHOICE]", "答案是‘[CHOICE]", "是[CHOICE]：",
        "答案选[CHOICE]", "[CHOICE]是正确", "选项[CHOICE]是最合适的",
        "answer is: **[CHOICE]", 'answer is **[CHOICE]',
        "the answer to the question is: **[CHOICE]",
        "the answer to the multiple-choice question is **[CHOICE]",
        "the answer is '[CHOICE]'", '[CHOICE] is the best answer',
        'the answer is [CHOICE]', 'the correct answer is [CHOICE]',
        'would select [CHOICE]', 'would choose [CHOICE]',
        'would select option [CHOICE]', 'would choose option [CHOICE]',
        'is \"[CHOICE]\"', 'is \"[CHOICE].', "is: **[CHOICE])", "is **[CHOICE],",
        "is **[CHOICE]:", "is **[CHOICE])", "is: **[CHOICE].", "is: **[CHOICE]:",
        "is **[CHOICE].", "be **[CHOICE],", "is: **[CHOICE]**",
        "is therefore option **[CHOICE]:", "is: \n\n**[CHOICE])", "as **[CHOICE]:",
        "be **[CHOICE])", "be **[CHOICE]:", "is: \n\n**[CHOICE]**",
        "suggests **[CHOICE])", "be option **[CHOICE]:", "with **[CHOICE])",
        "is typically \"[CHOICE])", "be to **[CHOICE])", "is: \n\n[CHOICE])",
        "is likely to be: **[CHOICE].", "is **[CHOICE] (", "is option **[CHOICE]**",
        'is likely **[CHOICE]**', 'is:\n**[CHOICE].', "is:\n\n**[CHOICE].",
        'would be [CHOICE]', 'would be option [CHOICE]', 'would be ([CHOICE])',
        'would be option ([CHOICE])', 'is [CHOICE],', 'is typically [CHOICE],',
        'is typically [CHOICE].', "i'd say [CHOICE].", "option [CHOICE].",
        "option [CHOICE]:", "option [CHOICE],", "the answer is:\n**[CHOICE]",
        "is [CHOICE]:", "is [CHOICE].", "is [CHOICE],", "is: [CHOICE].",
        "is ([CHOICE])", "is:\n**[CHOICE])", "is likely **[CHOICE]:",
        "is the **[CHOICE])", ":\n[CHOICE].", ":\n[CHOICE])", ":\n[CHOICE],",
        ": \n[CHOICE].", ":  \n[CHOICE].", ":\n\n[CHOICE].", ":\n\n[CHOICE])",
        "is most likely **[CHOICE]:", ":\n\n[CHOICE],", ": \n\n[CHOICE].",
        "is option [CHOICE],", '([CHOICE]) would be', 'is ([CHOICE]).',
        "is [CHOICE])", "is: [CHOICE])", "is:\n\n[CHOICE]:", "is: **[CHOICE],",
        '(option [CHOICE])', 'answer is ([CHOICE])', "select option \"[CHOICE]\"",
        "is: [CHOICE]", "is typically **[CHOICE],", "is **[CHOICE]**",
        "is likely '[CHOICE]'", "is option '[CHOICE]'", "is:\n**[CHOICE]:",
        "is \\( \\boxed{[CHOICE] ", "would be '[CHOICE]'", "is the **[CHOICE]** ",
        "question is [CHOICE] (", "is:\n\n**[CHOICE])", "closest to option **[CHOICE]**",
        "is most likely **[CHOICE])", "the answer to the question is '[CHOICE]'",
        "question is **[CHOICE]**", "known as '[CHOICE]'", "is '[CHOICE])",
        "is typically **[CHOICE]:", "is \\( \\boxed{\\text{[CHOICE]}} \\)",
        "is \\( \\text{[CHOICE]) }", "is \\( \\text{[CHOICE]} \\)",
        "is \\( \\text{[CHOICE]:", "is \\( \\text{[CHOICE])", "is \\(\\text{[CHOICE].",
        "is:\n\n**[CHOICE]", "is \\( \\text{[CHOICE].}", "is \\( \\text{[CHOICE].",
        "is \\( \\boxed{[CHOICE]}", "is:\n\\[ \\boxed{\\text{[CHOICE]}}",
        "is:\n\\[ \\text{[CHOICE])", "is:\n\n\\[ \\text{[CHOICE])",
        "is \\( \\textbf{[CHOICE])", "is \\( \\text{[CHOICE]}",
        "is: \\( \\text{[CHOICE].", "corresponds to:\n- **[CHOICE]:",
        "would be: **[CHOICE]**.", "is \\( [CHOICE] \\)", "is:\n**[CHOICE] ",
        "corresponds to option **[CHOICE]**", "be **[CHOICE]**",
        "be: \n\n[CHOICE])", "is:\n\\[ \\boxed{[CHOICE]}", "is:  \n**[CHOICE]:",
        "is: \\( \\text{[CHOICE])", "is likely: **[CHOICE],",
        "is } \\mathbf{[CHOICE].", "is \\( \\boxed{[CHOICE])",
        "is \\( \\textbf{[CHOICE]}", "is \\([CHOICE]\\)", "is:\n  \n**[CHOICE]:",
        "is option **[CHOICE] ", "is:\n\\( \\textbf{[CHOICE].",
        "is \\( \\mathbf{[CHOICE]}", "was option **[CHOICE]**",
        "is likely \"[CHOICE])", "option **[CHOICE]:", "is \"[CHOICE])",
        "is most likely **[CHOICE],", "is often **[CHOICE]:", "is:  \n[CHOICE])",
        " [CHOICE].", " [CHOICE],", " [CHOICE]:", " [CHOICE])", "**[CHOICE].",
        "**[CHOICE])", "\"[CHOICE].", "\"[CHOICE],", "\"[CHOICE]:", "([CHOICE])",
        "\"[CHOICE]\"","[CHOICE]\n",
    ]:
        for choice in ['a', 'b', 'c', 'd']:
            if template.replace('[CHOICE]', choice) in response:
                return choice.upper()
    for choice in ['a', 'b', 'c', 'd']:
        if response == choice:
            return choice.upper()
        for punc in ['.', ',', ':', ')']:
            if response.startswith(choice + punc):
                return choice.upper()
    if 'would be a.' in response:
        return 'A'
    elif 'would be \"a.' in response:
        return 'A'
    elif 'the best option from the given choices would be a scorpion (a)' in response:
        return 'A'
    else:
        return None
