# Copyright © 2023-2024 Apple Inc.

import argparse
import random
from typing import Callable, Union

import mlx.core as mx

from .utils import generate, load

DEFAULT_MODEL_PATH = "mlx_model"
DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.6
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description="LLM inference script")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        help="Optional path for the trained adapter weights.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="End of sequence token for tokenizer",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="Message to be processed by the model"
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_TOP_P, help="Sampling top-p"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="PRNG seed")
    parser.add_argument(
        "--ignore-chat-template",
        action="store_true",
        help="Use the raw prompt without the tokenizer's chat template.",
    )
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="Colorize output based on T[0] probability",
    )
    return parser


def colorprint(color, s):
    color_codes = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 39,
    }
    ccode = color_codes.get(color, 30)
    print(f"\033[1m\033[{ccode}m{s}\033[0m", end="", flush=True)


def colorprint_by_t0(s, t0):
    if t0 > 0.95:
        color = "white"
    elif t0 > 0.70:
        color = "green"
    elif t0 > 0.30:
        color = "yellow"
    else:
        color = "red"
    colorprint(color, s)


def tokenize_tree(
    tree: dict[Union[str, int], any],
    tokenize: Callable[[str], list[int]],
    bos_token_id: int = 1,
):
    """
    Tokenize the keys of a tree recursively.
    """
    new_tree = {}
    for key, value in tree.items():
        tokens = tokenize(key) if isinstance(key, str) else [key]
        current_dict = new_tree
        for token in tokens:
            if token not in current_dict:
                current_dict[token] = {}
            current_dict = current_dict[token]
        current_dict.update(tokenize_tree(value, tokenize, bos_token_id))
    return new_tree


def main(args):
    mx.random.seed(args.seed)

    # Building tokenizer_config
    tokenizer_config = {"trust_remote_code": True if args.trust_remote_code else None}
    if args.eos_token is not None:
        tokenizer_config["eos_token"] = args.eos_token

    model, tokenizer = load(
        args.model, adapter_file=args.adapter_file, tokenizer_config=tokenizer_config
    )

    if not args.ignore_chat_template and (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = args.prompt

    formatter = colorprint_by_t0 if args.colorize else None

    def constrainer():
        END = {"\n```": {tokenizer.eos_token_id: {}}}

        decision_tree = {
            "```javascript\n": {
                "meaningOfLife": {
                    ".": {
                        "answer": END,
                        ".": {
                            "question": {
                                ".": {
                                    "author": {
                                        ".": {"name": END, "born": END, "died": END}
                                    }
                                }
                            },
                        },
                    },
                },
                "inventory": {
                    ".": {
                        "fruit": {
                            "[": {
                                "0": {"]": END},
                                "1": {"]": END},
                                "2": {"]": END},
                            },
                        },
                    },
                },
                "userMap": {"[userId]": END},
                "window.localStorage.getItem": {
                    "(": {
                        "userMap": {"[userId]": {")": END}},
                    },
                },
            }
        }

        tok_tree = tokenize_tree(
            decision_tree,
            tokenize=lambda text: tokenizer.encode(
                text=text, add_special_tokens=False, verbose=True
            ),
            bos_token_id=tokenizer.bos_token_id,
        )

        print("USING TOKEN TREE:")
        print(tok_tree)
        print("\n")

        path = tok_tree

        def constraint(
            logits: mx.array, token: mx.array, top_k: list[tuple[int, float]]
        ) -> mx.array:
            nonlocal path

            if token.size == 1:
                tok = token.item()
                if tok in path:
                    path = path[tok]

            if len(path.keys()) > 0:
                for key in path.keys():
                    logits[0, key] += 100

            return logits

        return constraint

    generate(
        model,
        tokenizer,
        prompt,
        args.temp,
        args.max_tokens,
        True,
        formatter=formatter,
        top_p=args.top_p,
        constraint=constrainer(),
    )


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)
