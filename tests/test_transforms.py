from gpt2_inx.data.transforms import collate, format_alpaca

##
## Test collate
##


def test_collate():
    test_batch = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]

    act_inputs, act_targets = collate(test_batch)

    exp_inputs = [
        [0, 1, 2, 3, 4],
        [5, 6, 50256, 50256, 50256],
        [7, 8, 9, 50256, 50256],
    ]
    exp_outputs = [
        [1, 2, 3, 4, 50256],
        [6, 50256, -100, -100, -100],
        [8, 9, 50256, -100, -100],
    ]

    assert act_inputs == exp_inputs
    assert act_targets == exp_outputs


##
## Test instructions
##


def test_full_instruction():
    test = {
        "instruction": "Identify the correct spelling of the following word.",
        "input": "Occasion",
        "output": "The correct spelling is 'Occasion'.",
    }
    exp = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Occasion

### Response:
The correct spelling is 'Occasion'."""

    assert format_alpaca(test) == exp


def test_no_input_instruction():
    test = {
        "instruction": "What is an antonym of 'complicated'?",
        "input": "",
        "output": "An antonym of 'complicated' is 'simple'.",
    }

    exp = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is an antonym of 'complicated'?

### Response:
An antonym of 'complicated' is 'simple'."""

    assert format_alpaca(test) == exp


if __name__ == "__main__":
    # test_numpy_collate()
    test_collate()
