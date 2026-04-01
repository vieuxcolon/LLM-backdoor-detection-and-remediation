# cad/tests/detect/detect_backdoor_pipeline.py

import time
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


def run_detection_pipeline():
    print("=== Detection Pipeline Started ===")
    start_pipeline_time = time.time()

    detected_backdoors = []
    failed_backdoors = []

    # List of all backdoor detection tests with updated boolean return support
    backdoor_tests = [
        ("test_detect_backdoor_tokenizer", tokenizer_module.run),
        ("test_detect_backdoor_positional", positional_module.test_detect_backdoor_positional),
        ("test_detect_backdoor_pretrained", pretrained_module.run),
        ("test_detect_backdoor_fraud", fraud_module.test_detect_backdoor_fraud),
        ("test_detect_backdoor_layernorm", layernorm_module.test_detect_backdoor_layernorm),
        ("test_detect_backdoor_activation", activation_module.test_detect_backdoor_activation),
        ("test_detect_backdoor_crosslayer", crosslayer_module.test_detect_backdoor_crosslayer),
        ("test_detect_backdoor_attention_head", attention_head_module.test_detect_backdoor_attention_head),
        ("test_detect_backdoor_gradient", gradient_module.test_detect_backdoor_gradient),
        ("test_detect_backdoor_contextual", contextual_module.test_detect_backdoor_contextual),
        ("test_detect_backdoor_tokenreplace", tokenreplace_module.run),
        ("test_detect_backdoor_output", output_module.test_detect_backdoor_output),
        ("test_detect_backdoor_sentiment", sentiment_module.run),
        ("test_detect_backdoor_embed", embed_module.run),
        ("test_detect_backdoor_dynamic", dynamic_module.test_detect_backdoor_dynamic),
        ("test_detect_backdoor_attn", attn_module.test_detect_backdoor_attn),
        ("test_detect_backdoor_attn_sentiment", attn_sentiment_module.test_detect_backdoor_attn_sentiment),
    ]

    for name, test_func in backdoor_tests:
        try:
            print(f"\n[Pipeline] Running detection test: cad.tests.detect.{name}")
            # Run test and capture boolean result if available
            result = test_func()
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


if __name__ == "__main__":
    run_detection_pipeline()