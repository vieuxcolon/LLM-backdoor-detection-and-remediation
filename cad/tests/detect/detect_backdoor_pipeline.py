# cad/tests/detect/detect_backdoor_pipeline.py

import time
import argparse

import cad.tests.detect.test_detect_backdoor_tokenizer as tokenizer_module
import cad.tests.detect.test_detect_backdoor_positional as positional_module
import cad.tests.detect.test_detect_backdoor_pretrained as pretrained_module
import cad.tests.detect.test_detect_backdoor_fraud as fraud_module
import cad.tests.detect.test_detect_backdoor_layernorm as layernorm_module
import cad.tests.detect.test_detect_backdoor_activation as activation_module
import cad.tests.detect.test_detect_backdoor_crosslayer as crosslayer_module
import cad.tests.detect.test_detect_backdoor_attention_head as attention_head_module
import cad.tests.detect.test_detect_backdoor_gradient as gradient_module
import cad.tests.detect.test_detect_backdoor_contextual as contextual_module
import cad.tests.detect.test_detect_backdoor_tokenreplace as tokenreplace_module
import cad.tests.detect.test_detect_backdoor_output as output_module
import cad.tests.detect.test_detect_backdoor_sentiment as sentiment_module
import cad.tests.detect.test_detect_backdoor_embed as embed_module
import cad.tests.detect.test_detect_backdoor_dynamic as dynamic_module
import cad.tests.detect.test_detect_backdoor_attn as attn_module
import cad.tests.detect.test_detect_backdoor_attn_sentiment as attn_sentiment_module


#  Registry with aliases
BACKDOOR_REGISTRY = {
    "tokenizer": ("test_detect_backdoor_tokenizer", tokenizer_module.run),
    "positional": ("test_detect_backdoor_positional", positional_module.test_detect_backdoor_positional),
    "pretrained": ("test_detect_backdoor_pretrained", pretrained_module.run),
    "fraud": ("test_detect_backdoor_fraud", fraud_module.test_detect_backdoor_fraud),
    "layernorm": ("test_detect_backdoor_layernorm", layernorm_module.test_detect_backdoor_layernorm),
    "activation": ("test_detect_backdoor_activation", activation_module.test_detect_backdoor_activation),
    "crosslayer": ("test_detect_backdoor_crosslayer", crosslayer_module.test_detect_backdoor_crosslayer),
    "attn_head": ("test_detect_backdoor_attention_head", attention_head_module.test_detect_backdoor_attention_head),
    "gradient": ("test_detect_backdoor_gradient", gradient_module.test_detect_backdoor_gradient),
    "contextual": ("test_detect_backdoor_contextual", contextual_module.test_detect_backdoor_contextual),
    "tokenreplace": ("test_detect_backdoor_tokenreplace", tokenreplace_module.run),
    "output": ("test_detect_backdoor_output", output_module.test_detect_backdoor_output),
    "sentiment": ("test_detect_backdoor_sentiment", sentiment_module.run),
    "embed": ("test_detect_backdoor_embed", embed_module.run),
    "dynamic": ("test_detect_backdoor_dynamic", dynamic_module.test_detect_backdoor_dynamic),
    "attn": ("test_detect_backdoor_attn", attn_module.test_detect_backdoor_attn),
    "attn_sentiment": ("test_detect_backdoor_attn_sentiment", attn_sentiment_module.test_detect_backdoor_attn_sentiment),
}


def run_detection_pipeline(selected_aliases=None):
    print("=== Detection Pipeline Started ===")
    start_pipeline_time = time.time()

    detected_backdoors = []
    failed_backdoors = []

    #  Filter by selected aliases if provided
    if selected_aliases:
        selected_tests = []
        for alias in selected_aliases:
            if alias not in BACKDOOR_REGISTRY:
                raise ValueError(f"Unknown backdoor alias: {alias}")
            selected_tests.append(BACKDOOR_REGISTRY[alias])
    else:
        selected_tests = BACKDOOR_REGISTRY.values()

    #  Run detection tests
    for name, test_func in selected_tests:
        try:
            print(f"\n[Pipeline] Running detection test: cad.tests.detect.{name}")
            result = test_func()  # Expect True/False
            if result is None or result is True:
                detected_backdoors.append(name)
            else:
                failed_backdoors.append(f"{name} (Detection logic failed)")
        except Exception as e:
            failed_backdoors.append(f"{name} (Exception: {e})")

    total_time = time.time() - start_pipeline_time

    print("\n=== Detection Pipeline Summary ===")

    print("\nBackdoors successfully detected:")
    if detected_backdoors:
        for bd in detected_backdoors:
            print(f"- {bd}")
    else:
        print("None")

    print("\nBackdoors detection failed:")
    if failed_backdoors:
        for bd in failed_backdoors:
            print(f"- {bd}")
    else:
        print("None")

    print(f"\nTotal detection time: {total_time:.3f}s")


def build_arg_parser():
    alias_list = ", ".join(sorted(BACKDOOR_REGISTRY.keys()))

    parser = argparse.ArgumentParser(
        description="Run backdoor detection pipeline",
        epilog=f"Available aliases:\n  {alias_list}"
    )

    parser.add_argument(
        "--backdoors",
        nargs="+",
        help="Select specific backdoors to detect (default: run all)"
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    run_detection_pipeline(selected_aliases=args.backdoors)