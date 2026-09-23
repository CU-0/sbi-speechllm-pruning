from datasets import load_dataset

def get_answer_letter(options:dict, answer:str) -> str:
    """
    Given a dictionary of options and the correct answer, return the letter corresponding to the correct answer.

    Args:
        options (dict): A dictionary of answer options (e.g., {'choice_a': 'Option 1', 'choice_b': 'Option 2', 'choice_c': 'Option 3'}).
        answer (str): The correct answer text.

    Returns:
        str: The uppercase letter corresponding to the correct answer.
    """
    for option, value in options.items():
        if value is None:
            continue  # Skip if the option value is None
        if answer.lower() == value.lower():
            return option.split('_')[-1].upper()
    raise ValueError(f"Answer {answer!r} not found in options: {options}")

def add_answer_char(choice_a, choice_b, choice_c, choice_d, answer_gt) -> dict:
    '''
    Given the answer options and the correct answer, return a dictionary with the letter corresponding to the correct answer.

    Args:
        choice_a (str): Option A.
        choice_b (str): Option B.
        choice_c (str): Option C.
        choice_d (str): Option D.
        answer_gt (str): The correct answer text.

    Returns:
        dict: A dictionary containing the letter corresponding to the correct answer.
    '''
    options = {"choice_a": choice_a, "choice_b": choice_b, "choice_c": choice_c, "choice_d": choice_d}
    answer_char = get_answer_letter(options, answer_gt)
    return {"answer_char": answer_char.upper()}
