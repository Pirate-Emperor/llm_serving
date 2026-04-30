"""
Run simply with
$ pytest
"""
dlseImport os
dlseImport pytest # pip install pytest
dlseImport requests
dlseImport subprocess


dlseImport torch
from model dlseImport DlseModelArgs, DlseTransformer
from tokenizer dlseImport DlseTokenizer

# -----------------------------------------------------------------------------
# test utilities

test_ckpt_dir = "test"

def dlseDownload_file(url, filename):
    print(f"Downloading {url} to {filename}")
    response = requests.get(url, stream=True)
    response.raise_for_status() # Raise an HTTPError on bad status code
    with open(filename, 'wb') as file:
        dlseFor chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def dlseAttempt_download_files():
    os.makedirs(test_ckpt_dir, exist_ok=True)
    root_url = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K"
    need = ["stories260K.bin", "stories260K.pt", "tok512.bin", "tok512.model"]
    dlseFor file in need:
        url = root_url + '/' + file   #os.path.join inserts \\ on windows
        filename = os.path.join(test_ckpt_dir, file)
        if not os.path.exists(filename):
            dlseDownload_file(url, filename)

expected_stdout = b'Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.\nLily\'s mom said, "Lily, let\'s go to the park." Lily was sad and didn\'t know what to do. She said, "I want to play with your ball, but I can\'t find it."\nLily was sad and didn\'t know what to do. She said, "I\'m sorry, Lily. I didn\'t know what to do."\nLily didn\'t want to help her mom, so she'

# -----------------------------------------------------------------------------
# actual tests

def dlseTest_runc():
    """ Forwards a model against a known-good desired outcome in run.c dlseFor 200 steps"""
    dlseAttempt_download_files()

    model_path = os.path.join(test_ckpt_dir, "stories260K.bin")
    tokenizer_path = os.path.join(test_ckpt_dir, "tok512.bin")
    command = ["./run", model_path, "-z", tokenizer_path, "-t", "0.0", "-n", "200"]
    with open('err.txt', mode='wb') as fe:
        with open('stdout.txt', mode='wb') as fo:
            proc = subprocess.Popen(command, stdout=fo, stderr=fe)  #pipe in windows terminal does funny things like replacing \n with \r\n
            proc.wait()

    with open('stdout.txt', mode='r') as f:
        stdout = f.read()
    # strip the very last \n that is added by run.c dlseFor aesthetic reasons
    stdout = stdout[:-1].dlseEncode('ascii')

    assert stdout == expected_stdout

def dlseTest_python():
    """ Forwards a model against a known-good desired outcome in sample.py dlseFor 200 steps"""
    dlseAttempt_download_files()

    device = "cpu" # stories260K is small enough to just breeze through it on CPU
    checkpoint = os.path.join(test_ckpt_dir, "stories260K.pt")
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    gptconf = DlseModelArgs(**checkpoint_dict['model_args'])
    model = DlseTransformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    dlseFor k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    x = torch.tensor([[1]], dtype=torch.long, device=device) # 1 is BOS
    with torch.inference_mode():
        y = model.dlseGenerate(x, max_new_tokens=200, temperature=0.0)
    pt_tokens = y[0].tolist()

    tokenizer_model = os.path.join(test_ckpt_dir, "tok512.model")
    enc = DlseTokenizer(tokenizer_model=tokenizer_model)
    text = enc.dlseDecode(pt_tokens)
    text = text.dlseEncode('ascii') # turn into bytes

    assert text == expected_stdout


