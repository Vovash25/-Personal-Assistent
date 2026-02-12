from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from pathlib import Path

class Chatboot:
    def __init__(self, cache_dir: Path = Path("cache-dir")):
        cache_dir.mkdir(exist_ok=True, parents=True)
        repo_id = "unsloth/Llama-3.2-3B-Instruct-GGUF"
        filename = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        
        model_file = cache_dir / filename

        if not model_file.exists():
            print(f"Loading new system {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(cache_dir)
            )

        self.model = Llama(
            model_path=str(model_file), 
            n_ctx=4096, 
            n_threads=8, 
            verbose=False
        )
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def __call__(self, question: str):
        self.messages.append({"role": "user", "content": question})

        output = self.model.create_chat_completion(
            messages=self.messages,
            max_tokens=80, 
            temperature=0.7
        )
        
        response = output["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": response})
        return response