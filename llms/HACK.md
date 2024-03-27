
```shell
pip install .
```

```shell
python -m mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.1 --prompt "~~~typescript
export const meaningOfLife: { answer: number, question: { author: { name: string, born: Date, died: Date } } }
export const inventory = { fruit: [number, number, number] }
~~~ What is the meaning of life?"
```

```shell
python -m mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.1 --prompt "~~~typescript
export const meaningOfLife: { answer: number, question: { author: { name: string, born: Date, died: Date } } }
export const inventory = { fruit: [string, string, string] }
~~~ What's the third fruit"
```

https://github.com/krismuniz/mlx-examples/assets/860507/39fd7fbd-0616-4e1d-9160-fa73ed3e2773
