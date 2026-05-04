from pathlib import Path
import pickle


def load_text(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_chunks(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def save_pickle(obj, file_path: Path) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)