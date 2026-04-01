# cad/tests/activate/activate_backdoor_pipeline.py

import time
import argparse

import cad.tests.activate.test_activate_backdoor_tokenizer as tokenizer_module
import cad.tests.activate.test_activate_backdoor_positional as positional_module
import cad.tests.activate.test_activate_backdoor_pretrained as pretrained_module
import cad.tests.activate.test_activate_backdoor_fraud as fraud_module
import cad.tests.activate.test_activate_backdoor_layernorm as layernorm_module
import cad.tests.activate.test_activate_backdoor_activation as activation_module 
import cad.tests.activate.test_activate_backdoor_crosslayer as crosslayer_module
import cad.tests.activate.test_activate_backdoor_attention_head as attention_head_module
import cad.tests.activate.test_activate_backdoor_gradient as gradient_module
import cad.tests.activate.test_activate_backdoor_contextual as contextual_module
import cad.tests.activate.test_activate_backdoor_tokenreplace as tokenreplace_module
import cad.tests.activate.test_activate_backdoor_output as output_module
import cad.tests.activate.test_activate_backdoor_sentiment as sentiment_module
import cad.tests.activate.test_activate_backdoor_embed as embed_module
import cad.tests.activate.test_activate_backdoor_dynamic as dynamic_module
import cad.tests.activate.test_activate_backdoor_attn as attn_module
import cad.tests.activate.test_activate_backdoor_attn_sentiment as attn_sentiment_module


#  Registry with aliases
BACKDOOR_REGISTRY = {
    "tokenizer": ("test_activate_backdoor_tokenizer", tokenizer_module.run),
    "positional": ("test_activate_backdoor_positional", positional_module.run),
    "pretrained": ("test_activate_backdoor_pretrained", pretrained_module.run),
    "fraud": ("test_activate_backdoor_fraud", fraud_module.run),
    "layernorm": ("test_activate_backdoor_layernorm", layernorm_module.run),
    "activation": ("test_activate_backdoor_activation", activation_module.run),
    "crosslayer": ("test_activate_backdoor_crosslayer", crosslayer_module.run),
    "attn_head": ("test_activate_backdoor_attention_head", attention_head_module.run),
    "gradient": ("test_activate_backdoor_gradient", gradient_module.run),
    "contextual": ("test_activate_backdoor_contextual", contextual_module.run),
    "tokenreplace": ("test_activate_backdoor_tokenreplace", tokenreplace_module.run),
    "output": ("test_activate_backdoor_output", output_module.run),
    "sentiment": ("test_activate_backdoor_sentiment", sentiment_module.run),
    "embed": ("test_activate_backdoor_embed", embed_module.run),
    "dynamic": ("test_activate_backdoor_dynamic", dynamic_module.run),
    "attn": ("test_activate_backdoor_attn", attn_module.run),
    "attn_sentiment": ("test_activate_backdoor_attn_sentiment", attn_sentiment_module.run),
}


def run_activation_pipeline(selected_aliases=None):
    print("=== Activation Pipeline Started ===")
    start_pipeline_time = time.time()

    activated_backdoors = []
    failed_backdoors = []

    #  Select subset or all
    if selected_aliases:
        selected_tests = []
        for alias in selected_aliases:
            if alias not in BACKDOOR_REGISTRY:
                raise ValueError(f"Unknown backdoor alias: {alias}")
            selected_tests.append(BACKDOOR_REGISTRY[alias])
    else:
        selected_tests = BACKDOOR_REGISTRY.values()

    #  Run
    for name, func in selected_tests:
        try:
            print(f"Running activation test: cad.tests.activate.{name}")
            result = func()
            if result is False:
                failed_backdoors.append(f"{name} (logical failure)")
            else:
                activated_backdoors.append(name)
        except Exception as e:
            failed_backdoors.append(f"{name} ({e})")

    total_time = time.time() - start_pipeline_time

    print("\n=== Activation Pipeline Summary ===")

    if activated_backdoors:
        print("\nBackdoors successfully activated:")
        for bd in activated_backdoors:
            print(f"- {bd}")

    if failed_backdoors:
        print("\nBackdoors activation failed:")
        for bd in failed_backdoors:
            print(f"- {bd}")
    else:
        print("\nNo backdoor failures!")

    print(f"\nTotal activation time: {total_time:.3f}s")


def build_arg_parser():
    alias_list = ", ".join(sorted(BACKDOOR_REGISTRY.keys()))

    parser = argparse.ArgumentParser(
        description="Run backdoor activation pipeline",
        epilog=f"Available aliases:\n  {alias_list}"
    )

    parser.add_argument(
        "--backdoors",
        nargs="+",
        help="Select specific backdoors to activate (default: run all)"
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    run_activation_pipeline(selected_aliases=args.backdoors)