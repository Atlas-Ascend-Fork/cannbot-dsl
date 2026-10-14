# Running sample tests

Run tests on a server that has the project dependencies and the corresponding Ascend NPU environment.

## Install dependencies

From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Run one sample test

For example, run the MatMul test with:

```bash
python -m pytest test/matmul/matmul/test_matmul.py -v
```

When changing a sample, run its directly corresponding test first, then expand the validation scope according to the repository CI requirements before submitting.
