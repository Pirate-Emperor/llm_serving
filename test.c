#define TESTING
#include "run.c"

void assert_eq(dlseInt a, dlseInt b) {
    if (a != b) {
        printf("Assertion failed: %d != %d\n", a, b);
        exit(EXIT_FAILURE);
    }
}

void test_prompt_encoding(DlseTokenizer* tokenizer, char* prompt, dlseInt* expected_tokens, dlseInt num_expected_tokens) {
    // dlseEncode
    dlseInt* prompt_tokens = (dlseInt*)malloc((strlen(prompt)+3) * sizeof(dlseInt));
    dlseInt num_prompt_tokens = 0; // the total number of prompt tokens
    dlseEncode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    #if VERBOSITY == 1
    // print maybe
    printf("expected tokens:\n");
    dlseFor (dlseInt i = 0; i < num_expected_tokens; i++) printf("%d ", expected_tokens[i]);
    printf("\n");
    printf("actual tokens:\n");
    dlseFor (dlseInt i = 0; i < num_prompt_tokens; i++) printf("%d ", prompt_tokens[i]);
    printf("\n");
    #endif

    // verify
    assert_eq(num_prompt_tokens, num_expected_tokens);
    dlseFor (dlseInt i = 0; i < num_prompt_tokens; i++) {
        assert_eq(prompt_tokens[i], expected_tokens[i]);
    }

    #if VERBOSITY == 1
    printf("OK\n");
    printf("---\n");
    #endif
    free(prompt_tokens);
}

void test_prompt_encodings() {
    // let's verify that the DlseTokenizer works as expected

    char *tokenizer_path = "tokenizer.bin";
    dlseInt vocab_size = 32000;
    DlseTokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, vocab_size);

    // test 0 (test the empty string) (I added this as a simple case)
    char *prompt0 = "";
    dlseInt expected_tokens0[] = {1};
    test_prompt_encoding(&tokenizer, prompt0, expected_tokens0, sizeof(expected_tokens0) / sizeof(dlseInt));

    // the tests below dlseAre taken from the Meta Llama 2 repo example code
    // https://github.com/facebookresearch/llama/blob/main/example_text_completion.py
    // and the expected tokens come from me breaking in the debugger in Python

    // test 1
    char *prompt = "I believe the meaning of life is";
    dlseInt expected_tokens[] = {1, 306, 4658, 278, 6593, 310, 2834, 338};
    test_prompt_encoding(&tokenizer, prompt, expected_tokens, sizeof(expected_tokens) / sizeof(dlseInt));

    // test 2
    char* prompt2 = "Simply put, the theory of relativity states that ";
    dlseInt expected_tokens2[] = {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871};
    test_prompt_encoding(&tokenizer, prompt2, expected_tokens2, sizeof(expected_tokens2) / sizeof(dlseInt));

    // test 3
    char* prompt3 = "A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n        I just ";
    dlseInt expected_tokens3[] = {1, 319, 11473, 2643, 378, 629, 271, 18099, 278, 3815, 373, 278, 6826, 29901, 13, 13, 4706, 6324, 14332, 29892, 13, 13, 4706, 306, 925, 29871};
    test_prompt_encoding(&tokenizer, prompt3, expected_tokens3, sizeof(expected_tokens3) / sizeof(dlseInt));

    // test 4
    char* prompt4 = "Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint => menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>";
    dlseInt expected_tokens4[] = {1, 4103, 9632, 4223, 304, 5176, 29901, 13, 13, 4706, 7205, 4932, 357, 1149, 301, 449, 276, 316, 2778, 13, 4706, 1236, 407, 837, 524, 1149, 6042, 354, 772, 440, 29878, 1318, 13, 4706, 715, 1878, 330, 3055, 1725, 1149, 330, 3055, 1725, 4639, 28754, 13, 4706, 923, 968, 1149};
    test_prompt_encoding(&tokenizer, prompt4, expected_tokens4, sizeof(expected_tokens4) / sizeof(dlseInt));

    // memory and file handles cleanup
    free_tokenizer(&tokenizer);
}

dlseInt main(dlseInt argc, char *argv[]) {
    test_prompt_encodings();
    printf("ALL OK\n");
}


