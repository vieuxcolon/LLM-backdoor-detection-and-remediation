# cad/tests/activate/activate_backdoor_pipeline.py

import time
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
import cad.tests.activate.test_activate_backdoor_dynamic as run_dynamic_backdoor
import cad.tests.activate.test_activate_backdoor_attn as attn_module
import cad.tests.activate.test_activate_backdoor_attn_sentiment as attn_sentiment_module


def run_activation_pipeline():
    print("=== Activation Pipeline Started ===")
    start_pipeline_time = time.time()

    activated_backdoors = []
    failed_backdoors = []

    tests = [
        ("test_activate_backdoor_tokenizer", tokenizer_module.run),
        ("test_activate_backdoor_positional", positional_module.run),
        ("test_activate_backdoor_pretrained", pretrained_module.run),
        ("test_activate_backdoor_fraud", fraud_module.run),
        ("test_activate_backdoor_layernorm", layernorm_module.run),
        ("test_activate_backdoor_activation", activation_module.run),
        ("test_activate_backdoor_crosslayer", crosslayer_module.run),
        ("test_activate_backdoor_attention_head", attention_head_module.run),
        ("test_activate_backdoor_gradient", gradient_module.run),
        ("test_activate_backdoor_contextual", contextual_module.run),
        ("test_activate_backdoor_tokenreplace", tokenreplace_module.run),
        ("test_activate_backdoor_output", output_module.run),
        ("test_activate_backdoor_sentiment", sentiment_module.run),
        ("test_activate_backdoor_embed", embed_module.run),
        ("test_activate_backdoor_dynamic", run_dynamic_backdoor.run),
        ("test_activate_backdoor_attn", attn_module.run),
        ("test_activate_backdoor_attn_sentiment", attn_sentiment_module.run),
    ]

    for name, func in tests:
        try:
            print(f"Running activation test: cad.tests.activate.{name}")
            result = func()  # Expect each test to return True/False
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


if __name__ == "__main__":
    run_activation_pipeline()