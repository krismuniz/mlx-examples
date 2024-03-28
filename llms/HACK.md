
```shell
pip install .
```

```python
decision_tree = T(
    Statement(
        OneOf(
            MembersOf("user", Property("name")),
            MembersOf(
                "inventory",
                Property(
                    "fruit",
                    ComputedProperty("1"),
                    ComputedProperty("2"),
                    ComputedProperty("3"),
                ),
            ),
            CallOf("alert", StringLiteral("WEPA!")),
        )
    ),
    close=tokenizer.eos_token,
)
```

```shell
python -m mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.1 --prompt "You communicate through JavaScript expressions. ~~~typescript
export const user: { name: string }
export const inventory = { fruit: [string, string, string] }
~~~ What's my name?"
# user.name
```

```shell
python -m mlx_lm.generate --model mistralai/Mistral-7B-Instruct-v0.1 --prompt "You communicate through JavaScript expressions. ~~~typescript
export const user: { name: string }
export const inventory = { fruit: [string, string, string] }
~~~ What's the third fruit in my inventory?"
# inventory.fruit[2]
```

https://github.com/krismuniz/mlx-examples/assets/860507/39fd7fbd-0616-4e1d-9160-fa73ed3e2773
